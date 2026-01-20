{
  description = "Mandelbrot Set Visualizer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.beta.latest.default;

        # 実行時に必要な依存関係
        runtimeDeps = with pkgs; [
          vulkan-loader
          wayland
          libxkbcommon
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          xorg.libXrandr
        ];

        # ビルド時に必要な依存関係
        buildDeps = with pkgs; [
          pkg-config
          openssl
        ];
      in
      {
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "mandelbrot_set";
          version = "0.1.0";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = buildDeps;
          buildInputs = runtimeDeps;

          # ランタイム時にライブラリを見つけられるようにする
          postFixup = ''
            patchelf --add-rpath ${pkgs.lib.makeLibraryPath runtimeDeps} $out/bin/mandelbrot_set
          '';
        };

        devShells.default = pkgs.mkShell {
          buildInputs = buildDeps ++ runtimeDeps ++ [ rustToolchain ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeDeps;
        };
      }
    );
}
