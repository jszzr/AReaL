# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for the train worker."""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="AReaL Train Worker")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=30000, help="Bind port")
    parser.add_argument(
        "--admin-api-key",
        default="areal-admin-key",
        help="Admin API key for privileged operations",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["debug", "info", "warning", "error"],
        help="Log level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Experiment base seed used to initialize the training engine",
    )
    parser.add_argument(
        "--seed-role",
        default=None,
        help="Logical training role used to derive the worker seed",
    )
    parser.add_argument(
        "--seed-rank",
        type=int,
        default=None,
        help="Global training rank used to derive the worker seed",
    )
    args, _ = parser.parse_known_args()

    from areal.infra.utils.http import validate_admin_api_key
    from areal.v2.training_service.worker.app import create_app
    from areal.v2.training_service.worker.config import TrainWorkerConfig

    validate_admin_api_key(args.host, args.admin_api_key)

    config = TrainWorkerConfig(
        host=args.host,
        port=args.port,
        admin_api_key=args.admin_api_key,
        log_level=args.log_level,
        seed=args.seed,
        seed_role=args.seed_role,
        seed_rank=args.seed_rank,
    )

    from areal.utils.logging import suppress_http_loggers

    suppress_http_loggers()

    app = create_app(config)
    app.run(host=config.host, port=config.port, threaded=True)


if __name__ == "__main__":
    main()
