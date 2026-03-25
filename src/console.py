import typer
from .commands import gee, lidar, storage

app = typer.Typer()

app.add_typer(gee.app, name="gee")
app.add_typer(storage.app, name="storage")
app.add_typer(lidar.app, name="lidar")

if __name__ == "__main__":
    app()
