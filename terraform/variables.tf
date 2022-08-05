
variable "project" {
  default = "curious-song-343314"
}

variable "credentials_file" {
  default = "../keys/terraform-keys.json"
}

variable "region" {
  default = "europe-west1"
}

variable "zone" {
  default = "europe-west1-b"
}

variable "development_machine_type" {
  default = "e2-standard-4"
}
variable "development_machine_disk_size" {
  default = "50"
}
variable "development_machine_ip" {
  default = " 10.132.0.2"
}