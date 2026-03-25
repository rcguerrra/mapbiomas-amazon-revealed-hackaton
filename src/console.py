import typer
from .commands import gee, storage

app = typer.Typer()

app.add_typer(gee.app, name="gee")
app.add_typer(storage.app, name="storage")

if __name__ == "__main__":
    app()
