## Glamm Apptainer Guide

This is a guide on how to use Apptainer to run Glamm model

### Install Apptainer
Run
```bash
curl -s https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh | bash -s - install-dir
```

In your local directory. The application can be found at `install-dir/bin/apptainer` then. You do not need sudo access to install it.

Important notice: if your host machine is beyond ubuntu 22.04, or the system glibc version is beyond 2.35, you may encounter problems when running the Apptainer. See [this Apptainer document](https://apptainer.org/docs/user/main/gpu.html) for details.

### Modify the .def and Config File
A `.def` file is similar to a Dockerfile, but it's for Apptainer. The biggest difference is that the image (`.sif` file) and its container instance created by Apptainer (most importantly, the `/opt` workspace folder) is strictly read-only, meaning that only mounted file can be written to (and persist).

For this reason, `HF_HOME` location (the place where your downloaded model goes) and output database file must be a mounted file system inside the container. By default, here is our setting:

 - HF_HOME: `/mnt/scratch/huggingface` (specified in the .def file)
 - output_db: `/mnt/scratch/glamm.db` (specified in glamm_apptainer.yaml)

If you make sure something on the host is mounted to `/mnt/scratch`, you can keep these default settings.

### Build Image
Run the following command:
```bash
apptainer build --fakeroot vlmlens_cuda11.sif vlmlens_cuda11.def
```

This may take some time.

### Run the Container
```bash
apptainer run --nv --bind [host path]:/mnt/scratch vlmlens_cuda11.sif
```

Change the `[host path]` to a location where you have full read/write access.
