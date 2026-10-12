# Penny Royal 3.0.1

This maintenance release fixes three startup problems reported after 3.0 and
adds the Penny Royal version to startup logs.

## Fixes

- **RAM PLE on WSL2.** Both PLE gather paths now use the device address returned
  by `cudaHostGetDevicePointer`. WSL2 can map registered host memory at a
  different address; passing the host address caused an illegal memory access
  on the first request. The address is resolved once per allocation, with
  cleanup if resolution fails. ([#64](https://github.com/jpezzulli/sglang-rtxpro6000/issues/64))
- **Persistent TileLang cache.** Compiled kernels now go under the configured
  cache directory, normally `/cache/tilelang` in the container. This fixes
  startup with custom user IDs and keeps the cache across container recreation.
  An explicit `TILELANG_CACHE_DIR` is preserved.
  ([#65](https://github.com/jpezzulli/sglang-rtxpro6000/issues/65))
- **NVMe PLE startup.** The reader's compatibility hashes now match the shipped
  source. The stale 2.5.3 hashes rejected the 3.0 attention backend before the
  model could load. Required hooks and mismatch checks remain enforced.
  ([#66](https://github.com/jpezzulli/sglang-rtxpro6000/issues/66))

## Startup version

The launchers now print `Penny Royal 3.0.1` and the source commit. Locally
modified source is marked in the same line, making bug reports easier to trace.

## Update

Native users can follow [BUILD.md](BUILD.md). Container users should pull
`ghcr.io/jpezzulli/sglang-rtxpro6000:v3.0.1` and restart with their existing
external configuration. Model and cache mounts stay in place. Fresh launch
files and the beta configurator select the new image.

The [3.0 release notes](RELEASE_NOTES_3.0.md) contain the feature list and
performance results. These fixes keep the same model and performance defaults.

## Thanks

- [alphastorm](https://github.com/alphastorm) — WSL2 reproduction, device-pointer
  fix and testing.
- [Matt Sapp](https://github.com/MattSapp) — custom-UID failure and persistent
  TileLang cache fix.
- [desva](https://github.com/desva) — NVMe PLE compatibility report.

Penny Royal implemented the fixes; Codex reviewed and checked the integration.
