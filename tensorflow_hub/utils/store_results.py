path = './results.txt'
file = open(path,'a+')


def store_results(name, num_of_classes, avg_score, time, video_type):
    string = "{};{};{};{};{}\n".format(name, num_of_classes, round(avg_score, 2), time, video_type)
    file.write(string)

# store_results('resnet', 100, 55, "155", "night")
# file.close()
# reader = open(path, 'r')
# for f in reader.readlines():
#   print(f.strip())