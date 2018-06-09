import argparse
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.misc import imsave


parser = argparse.ArgumentParser()

parser.add_argument('--input-path', type=str,
                    dest='input_path', help='Input article path',
                    required=True,)
parser.add_argument('--output-path', type=str,
                    dest='output_path', help='Output image path',
                    default='./result.png')
parser.add_argument('--contain-chinese', type=bool,
                    dest='contain_chinese', help='Whether contain Chinese', 
                    default=False)
parser.add_argument('--font-path', type=str,
                    dest='font_path', help='Font path',
                    default='/Users/wengao.ye/GitHub/deep_learning/data_set/simfang.ttf')
parser.add_argument('--width', type=int,
                    dest='width', help='Output image width', 
                    default=1600)
parser.add_argument('--height', type=int,
                    dest='height', help='Output image height', 
                    default=1200)
parser.add_argument('--dpi', type=int,
                    dest='dpi', help='Output image quality', 
                    default=600)

args = parser.parse_args()


def create_wordcloud():

    with open(args.input_path, 'r') as file:
        article = file.read()
        article = article.replace('\r\n', ' ')
        article = article.replace('\r', ' ')
        article = article.replace('\n', ' ')

    if args.contain_chinese:
        wordcloud = WordCloud(width=args.width,
                              height=args.height,
                              font_path=args.font_path,
                              collocations=False).generate(article)
    else:
        wordcloud = WordCloud(width=args.width,
                              height=args.height).generate(article)

    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

    imsave(args.output_path, wordcloud)


if __name__ == '__main__':
    create_wordcloud()

