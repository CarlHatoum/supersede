output "registry" {
  value = local.registry
}

output "api_url" {
  value = var.deploy_services ? google_cloud_run_v2_service.api[0].uri : null
}

output "llm_url" {
  value = var.deploy_services ? google_cloud_run_v2_service.llm[0].uri : null
}

output "build_service_account" {
  value = google_service_account.build.id
}
