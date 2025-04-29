#! /usr/bin/env bash

set -ex

caddy file-server --listen 0.0.0.0:3333 --browse
