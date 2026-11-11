# Two Cloud Run services:
#   supersede-api  FastAPI + MCP + A2A on CPU, public.
#   supersede-llm  Ollama with the model built in, one NVIDIA L4, scales to zero.
# The LLM service only accepts internal traffic: the API reaches it through Direct
# VPC egress, so it needs no token and nothing else on the internet can call it.

locals {
  registry  = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
  api_image = "${local.registry}/supersede-api:${var.image_tag}"
  llm_image = "${local.registry}/supersede-llm:${var.image_tag}"
  apis = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "billingbudgets.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  repository_id = "supersede"
  location      = var.region
  format        = "DOCKER"
  description   = "Supersede API and LLM images"

  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 3
    }
  }
  depends_on = [google_project_service.apis]
}

# --- Network: API -> LLM over the VPC ---------------------------------------

resource "google_compute_network" "vpc" {
  name                    = "supersede"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "run" {
  name                     = "supersede-run"
  network                  = google_compute_network.vpc.id
  region                   = var.region
  ip_cidr_range            = "10.8.0.0/24"
  private_ip_google_access = true # *.run.app is reached through Google's private path
}

# --- Service accounts -------------------------------------------------------

resource "google_service_account" "api" {
  account_id   = "supersede-api"
  display_name = "Supersede API (Cloud Run)"
}

resource "google_service_account" "llm" {
  account_id   = "supersede-llm"
  display_name = "Supersede LLM (Cloud Run, Ollama)"
}

resource "google_service_account" "build" {
  account_id   = "supersede-build"
  display_name = "Supersede image builds (Cloud Build)"
}

resource "google_project_iam_member" "build" {
  for_each = toset(["roles/artifactregistry.writer", "roles/logging.logWriter"])
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.build.email}"
}

# --- LLM: Ollama on an L4 ---------------------------------------------------

resource "google_cloud_run_v2_service" "llm" {
  count               = var.deploy_services ? 1 : 0
  name                = "supersede-llm"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_INTERNAL_ONLY"
  deletion_protection = false

  template {
    service_account                  = google_service_account.llm.email
    timeout                          = "600s"
    max_instance_request_concurrency = 4 # matches OLLAMA_NUM_PARALLEL
    gpu_zonal_redundancy_disabled    = true

    scaling {
      min_instance_count = 0
      max_instance_count = var.llm_max_instances
    }

    node_selector {
      accelerator = "nvidia-l4"
    }

    containers {
      image = local.llm_image
      ports {
        container_port = 8080
      }
      resources {
        limits = {
          cpu              = "4"
          memory           = "16Gi"
          "nvidia.com/gpu" = "1"
        }
        cpu_idle          = false # required with a GPU
        startup_cpu_boost = true
      }
      startup_probe {
        tcp_socket {
          port = 8080
        }
        initial_delay_seconds = 0
        period_seconds        = 5
        failure_threshold     = 60
        timeout_seconds       = 5
      }
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "llm_invoker" {
  count    = var.deploy_services ? 1 : 0
  name     = google_cloud_run_v2_service.llm[0].name
  location = var.region
  role     = "roles/run.invoker"
  # Ingress is internal-only, so only callers inside the VPC (the API) get through.
  member = "allUsers"
}

# --- API: FastAPI + MCP + A2A -----------------------------------------------

resource "google_cloud_run_v2_service" "api" {
  count               = var.deploy_services ? 1 : 0
  name                = "supersede-api"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account                  = google_service_account.api.email
    timeout                          = "300s" # a full-agent /assess takes minutes on CPU-side waits
    max_instance_request_concurrency = 20

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    vpc_access {
      egress = "ALL_TRAFFIC" # needed for *.run.app calls to count as internal
      network_interfaces {
        network    = google_compute_network.vpc.id
        subnetwork = google_compute_subnetwork.run.id
      }
    }

    containers {
      image = local.api_image
      ports {
        container_port = 8080
      }
      env {
        name  = "OLLAMA_BASE_URL"
        value = google_cloud_run_v2_service.llm[0].uri
      }
      env {
        name  = "SUPERSEDE_VARIANT"
        value = var.variant
      }
      env {
        name  = "HF_HUB_OFFLINE" # encoder is in the image; the VPC has no internet egress
        value = "1"
      }
      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }
      startup_probe {
        http_get {
          path = "/healthz"
        }
        period_seconds    = 5
        failure_threshold = 24
        timeout_seconds   = 3
      }
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "api_public" {
  count    = var.deploy_services ? 1 : 0
  name     = google_cloud_run_v2_service.api[0].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# --- Budget alert (optional) ------------------------------------------------

data "google_project" "this" {}

resource "google_billing_budget" "monthly" {
  count           = var.billing_account == null ? 0 : 1
  billing_account = var.billing_account
  display_name    = "supersede monthly"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }
  amount {
    specified_amount {
      currency_code = "EUR"
      units         = tostring(var.budget_eur)
    }
  }
  dynamic "threshold_rules" {
    for_each = [0.5, 0.9, 1.0]
    content {
      threshold_percent = threshold_rules.value
    }
  }
}
