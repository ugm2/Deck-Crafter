class LoggerWriter:
    """
    A logger utility class that redirects output to multiple writers, such as stdout and log files.
    """

    def __init__(self, *writers):
        """
        Initialize with the writers to which output will be redirected.

        :param writers: Writers such as sys.stdout, file objects, etc.
        """
        self.writers = writers

    def write(self, message: str):
        """
        Write a message to all writers and flush the buffers.

        :param message: The message to write.
        """
        for writer in self.writers:
            writer.write(message)
            writer.flush()

    def flush(self):
        """
        Flush all writers.
        """
        for writer in self.writers:
            writer.flush()
