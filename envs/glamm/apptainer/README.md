## Glamm Apptainer Guide

This is a guide on how to use Apptainer to run Glamm model.
It may require some effort setting up the environment, and please check [Frequently Asked Questions](#frequently-asked-questions) for common issues and solutions.

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


## Frequently Asked Questions
1. Q: There's an error message showing ``version `GLIBC_2.3x' not found``; how to resolve it?

   A: This error is likely from the mismatch between the host system's glibc version and the version expected by the container. The container expects `GLIBC_2.35`, and this error may show up if your host system has an **newer** version of glibc.

   To avoid the error, you can try downgrading the glibc version on your host system (sometimes, the host system supports flexible glibc versions through `module`).

   See more from the [Apptainer documentation](https://apptainer.org/docs/user/main/gpu.html).


2. Q: When installing apptainer, it shows the command `rpm2cpio` not found. How to resolve it?

    A: In this case, you will need to install rpm manually. Here is a quick script of commands for installing rpm-4.18.1:

    ```bash
    curl -O https://ftp.osuosl.org/pub/rpm/releases/rpm-4.18.x/rpm-4.18.1.tar.bz2
    tar -xjf rpm-4.18.1.tar.bz2 && cd rpm-4.18.1
    ./configure --prefix=/path/to/install  # replace this path with your preferred one
    make
    make install
    export PATH="/path/to/install/bin:$PATH"
    ```
