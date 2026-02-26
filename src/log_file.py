import logging
import sys
class Logger:
    def get_logs(acc):
        try:
            logger = logging.getLogger(acc)
            logger.setLevel(logging.DEBUG)

            handler = logging.FileHandler(f"D:\\Users\\geeth\\PycharmProjects\\Spam_Analysis\\spam_.csv\\main.log", mode='w')
            formate = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            handler.setFormatter(formate)
            logger.addHandler(handler)

            return logger

        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
        
            logger.info(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')