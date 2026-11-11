terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.11"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region

  # Budgets and some APIs need a quota project when using user credentials.
  billing_project       = var.project_id
  user_project_override = true
}
