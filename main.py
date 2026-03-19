from data_loader import load_all_annotations, plot_class_distribution


def main():
    df = load_all_annotations(split="train")
    plot_class_distribution(df, split="train")


if __name__ == "__main__":
    main()