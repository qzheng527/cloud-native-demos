#!/bin/bash

curr_dir=$(readlink -f $(dirname "${BASH_SOURCE[0]}"))
top_dir=$(dirname "${curr_dir}")

source ${curr_dir}/common.sh

version="87b8c67c7dc4cbc1fc2a516cbf60c015ad3feb4b"
md5value="2efe1fbfa80abb8e5ca08dcc52d5c87c"

download_metric_server() {
    cd ${top_dir}/.cache
    download_file ${top_dir}/.cache/metrics-server-${version}.zip \
        https://github.com/kubernetes-sigs/metrics-server/archive/${version}.zip \
        ${md5value}

    if [ ! -d ${top_dir}/.cache/metrics-server-${version} ]; then
        unzip metrics-server-${version}.zip
    fi
}

apply_patch() {
    if [ ! -f ${top_dir}/.cache/metrics-server-${version}.patch ]; then
        cat > ${top_dir}/.cache/metrics-server-${version}.patch << '_EOF'
diff --git a/deploy/kubernetes/metrics-server-deployment.yaml b/deploy/kubernetes/metrics-server-deployment.yaml
index e4bfeaf..d41da9f 100644
--- a/deploy/kubernetes/metrics-server-deployment.yaml
+++ b/deploy/kubernetes/metrics-server-deployment.yaml
@@ -33,6 +33,8 @@ spec:
         args:
           - --cert-dir=/tmp
           - --secure-port=4443
+          - --kubelet-insecure-tls=true
+          - --kubelet-preferred-address-types=InternalIP
         ports:
         - name: main-port
           containerPort: 4443

_EOF
        cd ${top_dir}/.cache/metrics-server-${version}
        patch -p1 < ${top_dir}/.cache/metrics-server-${version}.patch
    fi
}

install_yaml() {
    cd ${top_dir}/.cache/metrics-server-${version}
    kubectl apply -f deploy/kubernetes
}

mkdir -p ${top_dir}/.cache
download_metric_server
apply_patch
install_yaml