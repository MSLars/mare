import click
import srsly


@click.command()
@click.argument('file_path')
def main(file_path: str):

    data = srsly.read_json(file_path)

    strat: str
    for strat in data["strategies"]:

        print()

        print(f"EVALUATION RESULTS FOR {strat}")

        print()

        for attr in data["strategies"][strat]:
            if attr.endswith("labels") or "micro" not in attr :
                continue

            print(f"{attr}: {data['strategies'][strat][attr]}")


if __name__ == "__main__":
    main()