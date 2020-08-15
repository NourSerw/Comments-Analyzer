import logging


def service_func():
    logging.basicConfig(filename='commentsanalyzer_flask.log', filemode='a',
                        format='%(name)s - %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    log = logging.getLogger(__name__)


if __name__ == '__main__':
    service_func()
