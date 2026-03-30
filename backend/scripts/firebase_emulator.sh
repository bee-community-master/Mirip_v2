#!/usr/bin/env bash
set -euo pipefail

firebase emulators:start --only auth,firestore,storage
