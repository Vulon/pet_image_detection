terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "3.5.0"
    }
  }
}

provider "google" {
  credentials = file(var.credentials_file)

  project = var.project
  region  = var.region
  zone    = var.zone
}

resource "google_compute_instance" "default" {
  name         = "development-compute"
  machine_type = var.development_machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      size  = var.development_machine_disk_size
      type  = "pd-standard"
      image = "debian-10-buster-v20220719"
    }
  }



  network_interface {
    network    = "default"
    # network_ip = var.development_machine_ip
    access_config {
    }
  }


  service_account {
    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    email  = "627824484387-compute@developer.gserviceaccount.com"
    scopes = ["cloud-platform"]
  }

  provisioner "file" {
    source      = "Makefile"
    destination = "/home/night/Makefile"
  }
}