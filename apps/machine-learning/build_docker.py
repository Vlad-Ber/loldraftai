import os
import argparse


def build_and_push(tag: str, registry: str):
    """Build and push Docker image"""
    # Build image
    image_name = f"{registry}/league-model:{tag}"
    os.system(f"docker build -t {image_name} .")

    # Push to registry (skip if local)
    if registry != "local":
        os.system(f"docker push {image_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--registry", required=True, help="Docker registry URL")
    args = parser.parse_args()

    build_and_push(args.tag, args.registry)
