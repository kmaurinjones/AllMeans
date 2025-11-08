"""Command-line interface for AllMeans."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .api import TopicModel

app = typer.Typer(
    name="allmeans",
    help="Automatic topic discovery with zero LLMs, minimal input",
    no_args_is_help=True,
)
console = Console()


@app.command()
def fit(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="Input text file"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSON file for results"),
    ] = None,
    method: Annotated[
        str,
        typer.Option("--method", "-m", help="Clustering method"),
    ] = "kmeans",
    feature_method: Annotated[
        str,
        typer.Option("--features", "-f", help="Feature extraction method"),
    ] = "tfidf",
    n_clusters: Annotated[
        int | None,
        typer.Option("--clusters", "-k", help="Number of clusters (auto if not set)"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed progress and output"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress all output except errors"),
    ] = False,
) -> None:
    """Fit a topic model on input text."""
    # Validate flags
    if verbose and quiet:
        console.print("[red]Error:[/red] Cannot use --verbose and --quiet together")
        raise typer.Exit(1)

    # Read input
    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input_path}")
        raise typer.Exit(1)

    text = input_path.read_text(encoding="utf-8")

    if verbose:
        console.print(f"[cyan]Reading text from:[/cyan] {input_path}")
        console.print(f"[cyan]Text length:[/cyan] {len(text)} characters")

    # Fit model
    model = TopicModel(
        method=method,  # type: ignore[arg-type]
        feature_method=feature_method,  # type: ignore[arg-type]
        n_clusters=n_clusters,
        auto_k=(n_clusters is None),
        verbose=verbose,
    )

    model.fit(text)
    results = model.get_results()

    # Display results (unless quiet)
    if not quiet:
        console.print(f"\n[green]Found {len(results.topics)} topics![/green]\n")

    if not quiet:
        # Create table
        table = Table(title="Topics")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Label", style="magenta")
        table.add_column("Keywords", style="yellow")
        table.add_column("Size", justify="right", style="green")
        table.add_column("Coherence", justify="right", style="blue")

        for topic in results.topics:
            table.add_row(
                str(topic.id),
                topic.label,
                ", ".join(topic.keywords[:5]),
                str(topic.size),
                f"{topic.coherence:.3f}",
            )

        console.print(table)

        # Print metrics
        console.print("\n[cyan]Metrics:[/cyan]")
        for metric_name, value in results.scores.items():
            console.print(f"  {metric_name}: {value:.3f}")

    # Save output
    if output:
        output_data = {
            "topics": [
                {
                    "id": t.id,
                    "label": t.label,
                    "keywords": t.keywords,
                    "size": t.size,
                    "coherence": t.coherence,
                    "exemplars": t.exemplar_sentences,
                }
                for t in results.topics
            ],
            "metrics": results.scores,
            "config": results.config,
        }

        output.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
        if not quiet:
            console.print(f"\n[green]Results saved to:[/green] {output}")


@app.command()
def topics(
    results_path: Annotated[
        Path,
        typer.Option("--results", "-r", help="Results JSON file from fit"),
    ],
    topn: Annotated[
        int,
        typer.Option("--topn", "-n", help="Number of top topics to show"),
    ] = 10,
) -> None:
    """Display topics from results file."""
    if not results_path.exists():
        console.print(f"[red]Error:[/red] Results file not found: {results_path}")
        raise typer.Exit(1)

    data = json.loads(results_path.read_text(encoding="utf-8"))

    # Create table
    table = Table(title=f"Top {topn} Topics")
    table.add_column("ID", style="cyan")
    table.add_column("Label", style="magenta")
    table.add_column("Keywords", style="yellow")
    table.add_column("Size", justify="right", style="green")

    for topic in data["topics"][:topn]:
        table.add_row(
            str(topic["id"]),
            topic["label"],
            ", ".join(topic["keywords"][:5]),
            str(topic["size"]),
        )

    console.print(table)


@app.command()
def version() -> None:
    """Show AllMeans version."""
    from . import __version__

    console.print(f"AllMeans version {__version__}")


if __name__ == "__main__":
    app()
