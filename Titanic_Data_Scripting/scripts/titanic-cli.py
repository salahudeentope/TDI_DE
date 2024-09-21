from titanic import TitanicCleaner
import click
arg
from pathlib import Path



titanic = TitanicCleaner('./data/titanic.csv')

# @click.command("titanic")
# @click.version_option("0.1.0", prog_name="titanic")
# def titanic():
#     click.echo("Hello, Titanic!")

@click.command()
@click.version_option("0.1.0", prog_name="titanic")
@click.argument("path")
def cli(path):
    target_dir = Path(path)
    if not target_dir.exists():
        click.echo("The target directory doesn't exist")
        raise SystemExit(1)
    
    click.echo()
    
if __name__ == "__main__":
    cli()