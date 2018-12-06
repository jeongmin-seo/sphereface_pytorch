import os

data_root = "./data/code (2)/deep_learning_data_x"


def write_text(save_content, train_test_type):

    save_name = train_test_type + ".txt"

    def write_line(_save_content):
        for content in _save_content:
            f.write(content[0])
            f.write("\t")
            f.write(content[1])
            f.write("\n")

    if save_name in os.listdir('./'):
        with open(os.path.join("./", save_name), 'a') as f:
            write_line(save_content)

    else:
        with open(os.path.join("./", save_name), 'w') as f:
            write_line(save_content)


for data_kind in os.listdir(data_root):
    data_kind_path = os.path.join(data_root, data_kind)

    for variant_type in os.listdir(data_kind_path):
        variant_type_path = os.path.join(data_kind_path, variant_type)

        for train_test_type in os.listdir(variant_type_path):
            save_list = list()
            train_test_path = os.path.join(variant_type_path, train_test_type)

            for label in os.listdir(train_test_path):
                label_path = os.path.join(train_test_path, label)

                for data_name in os.listdir(label_path):
                    data_path = os.path.join(label_path, data_name)
                    save_list.append([data_path.replace(data_root, ""), str(int(label)-1)])

            write_text(save_list, train_test_type)

