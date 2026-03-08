#!/bin/bash
# Environment config for comparison visualization pipeline
# Modify these paths to your dataset locations
export LL_ROOT=$(echo L21udC9mL2RhdGFzZXRzL251c2NlbmVzX2xvd2xpZ2h0 | base64 -d)
export NM_ROOT=$(echo L21udC9mL2RhdGFzZXRzL251c2NlbmVz | base64 -d)
export NUSC_VERSION="v1.0-mini"
