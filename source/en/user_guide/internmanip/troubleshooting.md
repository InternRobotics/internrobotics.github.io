# ⚠️ Troubleshooting

## 1. Conda Terms of Service Acceptance Error

If you encounter the following error during the genmanip installation:
```bash
CondaToNonInteractiveError: Terms of Service have not been accepted for the following channels:
    • https://repo.anaconda.com/pkgs/main
    • https://repo.anaconda.com/pkgs/r
```
Manually accept the Terms of Service for each affected channel by running these commands:
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```


## 2. Tips for Slow or Unstable Networks

If you encounter errors such as timeouts or incomplete downloads, especially in network-restricted or low-bandwidth environments, we recommend the following approaches.

- Step 1: By default, `uv pip` uses relatively short HTTP timeouts. To extend the timeout, set the following environment variable before installation:
    ```bash
    export UV_HTTP_TIMEOUT=600  # Timeout in seconds (10 minutes)
    ```

- Step 2: Locate the package that failed to install. When running the `install.sh` script, if a package fails to install, identify the failing line in the output. This usually indicates which package wasn't downloaded properly.
- Step 3: Dry-run to preview packages and versions. Use uv pip install --dry-run` to preview which packages (and exact versions) are going to be installed. For example:
    ```bash
    uv pip install "git+https://github.com/NVIDIA/Isaac-GR00T.git#egg=isaac-gr00t[base]" --dry-run
    ```
    This will list all packages along with their resolved versions, including those that might fail due to slow download.
- Step 4: Activate your environment. Before manually installing the package, make sure you're in the correct virtual environment.
    ```bash
    source .venv/{your_env_name}/bin/activate
    ```
- Step 5: Manually install the problematic package. After identifying the package and version, install it manually:
    ```bash
    uv pip install torch==2.5.1  # Replace with your actual package and version
    ```

<!---
- By default, `uv pip` uses relatively short HTTP timeouts. To extend the timeout, set the following environment variable before installation:
    ```bash
    export UV_HTTP_TIMEOUT=600  # Timeout in seconds (10 minutes)
    ```
- To ensure successful installation without network interruptions, you can download some large packages first and then install them locally:
    ```bash
    uv pip download -d ./wheelhouse "some-large-package"
    uv pip install --no-index --find-links=./wheelhouse "some-large-package"
    ```
--->

## 3. `import simpler_env` Fails Due to Missing Vulkan Library

If you encounter the following error when trying to import simpler_env:
```bash
>>> import simpler_env
Traceback (most recent call last):
  ...
ImportError: libvulkan.so.1: cannot open shared object file: No such file or directory
```
You can resolve this issue by installing the Vulkan runtime library via `apt`:
```bash
sudo apt update
sudo apt install libvulkan1
sudo ldconfig
```




## 4. Failed to Install `pyzmq` When Building Model Dependency


When installing the `model` environment and encountering the following error:
```bash
Resolved 162 packages in 4.80s
× Failed to build `pyzmq==27.0.1`
├─▶ The build backend returned an error
╰─▶ Call to `scikit_build_core.build.build_wheel` failed (exit status: 1)
```
A recommended approach is to build from source by running:
```bash
./install.sh --model_bfs
```
If this fails with GCC-related errors, please refer to the troubleshooting advice below.

## 5. GCC Fails to Compile Due to Missing Dependencies

When compiling C++ components (e.g., `building ManiSkill2_real2sim` or `pyzmq`), you might encounter errors related to GCC or missing shared libraries. This guide walks you through how to resolve them without root/sudo permissions.

- Step 1: Use a modern GCC (recommended ≥ 9.3.0). Older system compilers (e.g., GCC 5.x or 7.x) may not support required C++ standards. It's recommended to switch to GCC 9.3.0 or newer:
    ```bash
    export LD_LIBRARY_PATH=${PATH_TO}/gcc/gcc-9.3.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export PATH=${PATH_TO}/gcc/gcc-9.3.0/bin:$PATH
    ```
    > ⚠️ Note: Simply using a newer compiler might not be enough — it may depend on shared libraries that are not available on your system.
- Step 2: Manually install required libraries. If you encounter errors like: `error while loading shared libraries: libmpc.so.2 (libmpfr.so.1, libgmp.so.3)`. If you do have `sudo` privileges, the easiest way is to install the required libraries system-wide using your system package manager.
    ```bash
    sudo apt update
    sudo apt install gcc-9 g++-9
    ```
    Or, you need to manually compile and install the following dependencies locally:
    ```bash
    INSTALL_DIR=$HOME/local
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    wget https://ftp.gnu.org/gnu/gmp/gmp-6.2.1.tar.xz
    tar -xf gmp-6.2.1.tar.xz && cd gmp-6.2.1
    ./configure --prefix="$INSTALL_DIR"
    make -j$(nproc)
    make install
    cd "$INSTALL_DIR"

    wget https://www.mpfr.org/mpfr-4.2.1/mpfr-4.2.1.tar.xz
    tar -xf mpfr-4.2.1.tar.xz && cd mpfr-4.2.1
    ./configure --prefix="$INSTALL_DIR" --with-gmp="$INSTALL_DIR"
    make -j$(nproc)
    make install
    cd "$INSTALL_DIR"

    echo "📦 Installing MPC..."
    wget https://ftp.gnu.org/gnu/mpc/mpc-1.3.1.tar.gz
    tar -xf mpc-1.3.1.tar.gz && cd mpc-1.3.1
    ./configure --prefix="$INSTALL_DIR" --with-gmp="$INSTALL_DIR" --with-mpfr="$INSTALL_DIR"
    make -j$(nproc)
    make install
    ```
- Step 3 (Optional): Fix Missing `.so` Versions. Sometimes you have the correct library version (e.g., `libgmp.so.10`), but GCC expects an older symlink name (e.g., `libgmp.so.3`). You can fix missing library versions with symlinks.
- Step 4: Export the Library Path. Make sure the compiler can find your locally installed shared libraries:
    ```bash
    export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
    ```
## 6. Handling `RuntimeError: vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent` in SimplerEnv Benchmark


This error is a tricky issue caused by a failed GPU driver installation when running the SimplerEnv benchmark. Many users have reported this problem in the [SimplerEnv GitHub issue #68](https://github.com/simpler-env/SimplerEnv/issues/68).

You may try the suggested solutions there, but they do not always work reliably. Therefore, we provide an alternative approach if you can use container images.

**Alternative Solution Using Apptainer Container**
- Make sure you have installed `simpler_env`.
- Install Apptainer on your system if you haven't already.
- Download a prebuilt container image with SimplerEnv installed from our [Hugging Face repository](https://huggingface.co/InternRobotics/Manishill2/tree/main).
- Run the evaluation inside the container using a command similar to the following:
    ```bash
    apptainer exec --bind /mnt:/mnt --nv path_to_maniskill2.sif path_to_your_python scripts/eval/start_evaluator.py --config run_configs/examples/internmanip_demo.py --server
    ```
    Replace path_to_maniskill2.sif with the actual path to the downloaded container image, and path_to_your_python with the Python interpreter you want to use inside the container.
