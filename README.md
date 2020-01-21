# prima-undine-opencl: OpenCL backend for Prima-undine ⛵

## Pre-requisites

* [CLBlast](https://github.com/CNugteren/CLBlast)

## Example

```rust
use prima_undine_opencl::OpenCL;

fn main() {
    let dev = OpenCL::new(0, 0);
    // ...
}
```
