# Air-Gapped GenAI Architecture Notes

## Context

Many regulated environments do not allow production clusters to pull images, models, or packages directly from the internet.

This pattern shows a safe public reference approach for preparing GenAI workloads for an isolated runtime environment.

## Key idea

The runtime cluster should not depend on public internet access.

Required artifacts should be prepared in a controlled staging zone, reviewed, scanned, mirrored, and then deployed from an internal registry.

## Main components

* Staging host with controlled internet access
* Private registry such as Harbor or private Azure Container Registry
* Air-gapped Kubernetes cluster
* Local model runtime
* Vector database
* RAG application or API layer
* Internal documentation source

## Design principles

1. No direct internet from runtime nodes
2. Pin image versions before deployment
3. Scan images before promoting them
4. Avoid embedding secrets in manifests
5. Keep configuration environment-specific
6. Document every exception and trade-off

## Trade-offs

### Benefits

* Better control over supply chain
* Reduced exposure to external dependencies
* Better fit for regulated environments
* Repeatable deployment process

### Limitations

* More operational effort
* Slower updates
* Image and model version management becomes critical
* Requires strong documentation and release discipline

## What this repo is not

This is not a full production deployment.

It does not include real banking architecture, private network information, customer data, or confidential configuration.

