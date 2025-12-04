import ast
import click
from pathlib import Path
from lab1.logic.predictor import predict_class, resize_image, get_image_size
from lab1.preprocessing import (
    remove_missing as remove_missing_func,
    fill_missing as fill_missing_func,
    remove_duplicates as remove_duplicates_func,
    normalize as normalize_func,
    standardize as standardize_func,
    clip as clip_func,
    convert_to_int as convert_to_int_func,
    log_transform as log_transform_func,
    tokenize_text as tokenize_text_func,
    remove_non_alnum_spaces as remove_non_alnum_spaces_func,
    remove_stopwords as remove_stopwords_func,
    flatten_list as flatten_list_func,
    shuffle_list as shuffle_list_func,
)


@click.group()
def cli():
    """Main CLI group for data preprocessing commands."""
    pass


# =====================================================================
# PREDICT AND RESIZE FROM THE DATA
# =====================================================================
@cli.command("predict")
@click.argument("image_path", type=click.Path(exists=True))
def predict(image_path):
    """Predict class for IMAGE_PATH (dummy random predictor)."""
    p = Path(image_path)
    with p.open("rb") as f:
        data = f.read()
    cls = predict_class(data)
    w, h = get_image_size(data)
    click.echo(f"predicted_class: {cls}")
    click.echo(f"original_size: {w}x{h}")


@cli.command("resize")
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("width", type=int)
@click.argument("height", type=int)
@click.option("-o", "--out", type=click.Path(), default=None, help="Output path")
def resize(image_path, width, height, out):
    """Resize image to WIDTH HEIGHT and save (overwrites if out exists)."""
    p = Path(image_path)
    out_path = Path(out) if out else p.with_name(f"{p.stem}_{width}x{height}{p.suffix}")
    with p.open("rb") as f:
        data = f.read()
    resized = resize_image(data, width, height)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resized)
    click.echo(f"Saved resized image to: {out_path}")


# =====================================================================
# CLEAN GROUP
# =====================================================================
@cli.group(help="Functions related to data cleaning")
def clean():
    pass


@clean.command(name="remove-missing", help="Remove missing values from a list")
@click.argument("values", nargs=-1)
def remove_missing(values):
    click.echo(remove_missing_func(list(values)))


@clean.command(name="fill-missing", help="Fill missing values in a list")
@click.argument("values", nargs=-1)
@click.option("--fill", default=0, help="Value used to fill missing entries")
def fill_missing(values, fill):
    click.echo(fill_missing_func(list(values), fill))


@clean.command(name="unique", help="Remove duplicate values")
@click.argument("values", nargs=-1)
def unique(values):
    click.echo(remove_duplicates_func(list(values)))


# =====================================================================
# NUMERIC GROUP
# =====================================================================
@cli.group(help="Functions related to numerical attributes")
def numeric():
    pass


@numeric.command(name="normalize")
@click.argument("values", nargs=-1, type=float)
@click.option("--min", "new_min", default=0.0, show_default=True)
@click.option("--max", "new_max", default=1.0, show_default=True)
def normalize(values, new_min, new_max):
    click.echo(normalize_func(list(values), new_min, new_max))


@numeric.command(name="standardize")
@click.argument("values", nargs=-1, type=float)
def standardize(values):
    click.echo(standardize_func(list(values)))


@numeric.command(name="clip")
@click.argument("values", nargs=-1, type=float)
@click.option("--min", "min_value", default=0.0, show_default=True)
@click.option("--max", "max_value", default=1.0, show_default=True)
def clip(values, min_value, max_value):
    click.echo(clip_func(list(values), min_value, max_value))


@numeric.command(name="convert-int")
@click.argument("values", nargs=-1)
def convert_int(values):
    click.echo(convert_to_int_func(list(values)))


@numeric.command(name="log")
@click.argument("values", nargs=-1, type=float)
def log(values):
    click.echo(log_transform_func(list(values)))


# =====================================================================
# TEXT GROUP
# =====================================================================
@cli.group(help="Functions for text processing")
def text():
    pass


@text.command(name="tokenize")
@click.argument("text")
def tokenize(text):
    click.echo(tokenize_text_func(text))


@text.command(name="remove-punct")
@click.argument("text")
def remove_punct(text):
    click.echo(remove_non_alnum_spaces_func(text))


@text.command(name="remove-stops")
@click.argument("text")
@click.option("--stopwords", multiple=True)
def remove_stops(text, stopwords):
    click.echo(remove_stopwords_func(text, list(stopwords)))


# =====================================================================
# STRUCT GROUP
# =====================================================================
@cli.group(help="Functions related to data structure")
def struct():
    pass


@struct.command(name="shuffle")
@click.argument("values", nargs=-1)
@click.option("--seed", type=int, default=None)
def shuffle(values, seed):
    click.echo(shuffle_list_func(list(values), seed))


@struct.command(name="flatten")
@click.argument("text")
def flatten(text):
    parsed = ast.literal_eval(text)
    click.echo(flatten_list_func(parsed))


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    cli()
