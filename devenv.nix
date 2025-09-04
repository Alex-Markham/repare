{ pkgs, lib, config, inputs, ... }:
let
  pkgs-unstable = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
in
{
  # https://devenv.sh/packages/
  packages = [
    pkgs.git

    # needed for snakemake
    pkgs.coinmp

    # needed for sempler DRFNet
    pkgs.R
    pkgs.rPackages.drf
    pkgs.zlib
    pkgs.bzip2
    pkgs.xz
    pkgs.zstd
    pkgs.libdeflate
    pkgs.icu
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = "3.13";
    venv.enable = true;
    uv = {
      enable = true;
      package = pkgs-unstable.uv;
      sync.enable = true;
    };
  };

  enterShell = ''
  '';

  # See full reference at https://devenv.sh/reference/options/
}
