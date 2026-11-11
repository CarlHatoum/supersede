variable "project_id" {
  description = "GCP project that hosts Supersede."
  type        = string
}

variable "region" {
  description = "Region with Cloud Run L4 GPUs (e.g. europe-west4, us-central1)."
  type        = string
  default     = "europe-west4"
}

variable "image_tag" {
  description = "Tag of the images built by deploy/cloudbuild.yaml."
  type        = string
  default     = "v1"
}

variable "deploy_services" {
  description = "false on the first apply (registry and network only, before images exist)."
  type        = bool
  default     = true
}

variable "variant" {
  description = "Model variant behind /assess: llm_only (one call) or full (agent with tools)."
  type        = string
  default     = "llm_only"
}

variable "llm_max_instances" {
  description = "Upper bound on GPU instances (each is one L4)."
  type        = number
  default     = 1
}

variable "billing_account" {
  description = "Billing account ID for the budget alert; null skips the budget."
  type        = string
  default     = null
}

variable "budget_eur" {
  description = "Monthly budget that triggers alert emails at 50/90/100%."
  type        = number
  default     = 30
}
