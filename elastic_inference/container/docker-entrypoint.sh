#!/bin/bash

CMD="$@"

source /opt/intel/openvino/bin/setupvars.sh
exec "$CMD"
