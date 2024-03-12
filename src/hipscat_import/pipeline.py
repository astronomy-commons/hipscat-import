"""Flow control and pipeline entry points."""

import smtplib
from email.message import EmailMessage

from dask.distributed import Client

import hipscat_import.catalog.run_import as catalog_runner
import hipscat_import.index.run_index as index_runner
import hipscat_import.margin_cache.margin_cache as margin_runner
import hipscat_import.soap.run_soap as soap_runner
import hipscat_import.verification.run_verification as verification_runner
from hipscat_import.catalog.arguments import ImportArguments
from hipscat_import.index.arguments import IndexArguments
from hipscat_import.margin_cache.margin_cache_arguments import MarginCacheArguments
from hipscat_import.runtime_arguments import RuntimeArguments
from hipscat_import.soap.arguments import SoapArguments
from hipscat_import.verification.arguments import VerificationArguments

# pragma: no cover


def pipeline(args: RuntimeArguments):
    """Pipeline that creates its own client from the provided runtime arguments"""
    with Client(
        local_directory=args.dask_tmp,
        n_workers=args.dask_n_workers,
        threads_per_worker=args.dask_threads_per_worker,
    ) as client:
        pipeline_with_client(args, client)


def pipeline_with_client(args: RuntimeArguments, client: Client):
    """Pipeline that is run using an existing client.

    This can be useful in tests, or when a dask client requires some more complex
    configuraion.
    """
    try:
        if not args:
            raise ValueError("args is required and should be subclass of RuntimeArguments")

        if isinstance(args, ImportArguments):
            catalog_runner.run(args, client)
        elif isinstance(args, IndexArguments):
            index_runner.run(args, client)
        elif isinstance(args, MarginCacheArguments):
            margin_runner.generate_margin_cache(args, client)
        elif isinstance(args, SoapArguments):
            soap_runner.run(args, client)
        elif isinstance(args, VerificationArguments):
            verification_runner.run(args)
        else:
            raise ValueError("unknown args type")

        _send_success_email(args)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        if args.completion_email_address:
            _send_failure_email(args, exception)
        raise exception


def _send_failure_email(args: RuntimeArguments, exception: Exception):
    message = EmailMessage()
    message["Subject"] = "hipscat-import failure."
    message["To"] = args.completion_email_address
    message.set_content(
        f"output_artifact_name: {args.output_artifact_name}"
        "\n\nSee logs for more details"
        f"\n\nFailed with message:\n\n{exception}"
    )

    _send_email(message)


def _send_success_email(args):
    if not args.completion_email_address:
        return
    message = EmailMessage()
    message["Subject"] = "hipscat-import success."
    message["To"] = args.completion_email_address
    message.set_content(f"output_artifact_name: {args.output_artifact_name}")

    _send_email(message)


def _send_email(message: EmailMessage):
    message["From"] = "updates@lsdb.io"

    # Send the message via our own SMTP server.
    with smtplib.SMTP("localhost") as server:
        server.send_message(message)
