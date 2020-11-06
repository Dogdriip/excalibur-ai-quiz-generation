import argparse
from TextExtractor import TextExtractor
from QuizGenerator import QuizGenerator

if __name__ == '__main__':
    # TODO: Clarify description.

    # Parse args.
    parser = argparse.ArgumentParser(
        description='Generate quiz from given pdf file.')
    parser.add_argument('file_path', type=str, help='file_path')
    # parser.add_argument('quiz_cnt', type=int, help='quiz_cnt')
    args = parser.parse_args()

    file_path = args.file_path
    # quiz_cnt = args.quiz_cnt

    # Extract text from pdf.
    extractor = TextExtractor(file_path)
    text = extractor.extract_text()

    # Generate quiz from text.
    quizgenerator = QuizGenerator(text)
    problem, answer = quizgenerator.generate_quiz()

    # Print `problem|output` to stdout.
    res = problem + "|" + answer
    print(res)
