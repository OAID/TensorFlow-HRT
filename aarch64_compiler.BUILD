package(default_visibility = ['//visibility:public'])

filegroup(
  name = 'gcc',
  srcs = [
    'usr/bin/aarch64-linux-gnu-gcc',
  ],
)

filegroup(
  name = 'ar',
  srcs = [
    'usr/bin/aarch64-linux-gnu-ar',
  ],
)

filegroup(
  name = 'ld',
  srcs = [
    'usr/bin/aarch64-linux-gnu-ld',
  ],
)

filegroup(
  name = 'nm',
  srcs = [
    'usr/bin/aarch64-linux-gnu-nm',
  ],
)

filegroup(
  name = 'objcopy',
  srcs = [
    'usr/bin/aarch64-linux-gnu-objcopy',
  ],
)

filegroup(
  name = 'objdump',
  srcs = [
    'usr/bin/aarch64-linux-gnu-objdump',
  ],
)

filegroup(
  name = 'strip',
  srcs = [
    'usr/bin/aarch64-linux-gnu-strip',
  ],
)

filegroup(
  name = 'as',
  srcs = [
    'usr/bin/aarch64-linux-gnu-as',
  ],
)

filegroup(
  name = 'compiler_pieces',
  srcs = glob([
    'usr/lib/gcc-cross/aarch64-linux-gnu/5/**',
    'usr/aarch64-linux-gnu/**',
  ]),
)

filegroup(
  name = 'compiler_components',
  srcs = [
    ':gcc',
    ':ar',
    ':ld',
    ':nm',
    ':objcopy',
    ':objdump',
    ':strip',
    ':as',
  ],
)
