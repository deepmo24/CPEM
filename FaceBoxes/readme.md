## How to fun FaceBoxes

### Build the cpu version of NMS
```shell script
cd utils
python3 build.py build_ext --inplace
```

or just run

```shell script
sh ./build_cpu_nms.sh
```

### Run the demo of face detection
```shell script
python3 FaceBoxes.py
```


### Build in Windows

1. In `utils/build.py`, comment the line `extra_compile_args=["-Wno-cpp", "-Wno-unused-function"]`.

2. Modify function `cpu_nms()` in `utils/nms/cpu_nms.pyx`.
 
    refer to https://github.com/cleardusk/3DDFA_V2/issues/12

3. run:

    ```shell script
    sh ./build_cpu_nms.sh
    ```