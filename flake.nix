{
  description = "Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python313
            uv
            # System libraries needed for NumPy and other scientific packages
            zlib
            wget
            gzip
            gcc
            stdenv.cc.cc.lib
          ];

          # Ensure libraries are findable at runtime
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];
          LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];

          shellHook = ''
            echo "Python development environment loaded"
            echo "Python: $(python --version)"
            echo "UV: $(uv --version)"
          '';
        };
      }
    );
}
