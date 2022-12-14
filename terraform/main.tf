terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "3.5.0"
    }
  }
}

resource "google_compute_address" "static" {
  name = "ipv4-address"
}

resource "google_storage_bucket" "static-site" {
  name          = var.mlflow_bucket
  location      = var.region
  force_destroy = false

  #uniform_bucket_level_access = true

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
    access_config {
      nat_ip = google_compute_address.static.address
    }
  }


  service_account {
    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    email  = "627824484387-compute@developer.gserviceaccount.com"
    scopes = ["cloud-platform"]
  }

  connection {
    type     = "ssh"
    user     = "night"
    host     = google_compute_address.static.address
    timeout  = "90s"
    private_key = file(var.development_machine_private_key)
  }
  provisioner "file" {
      source      = "../scripts/Makefile"
      destination = "/home/night/Makefile"
  }

provisioner "remote-exec" {
    script = "../scripts/Makefile"
  }

}
