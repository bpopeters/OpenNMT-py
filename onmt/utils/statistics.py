""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys
from itertools import chain

from torch.distributed import get_rank
from onmt.utils.distributed import all_gather_list
from onmt.utils.logging import logger


def accuracy(n_correct, n_words, **kwargs):
    return 100 * (n_correct / n_words)


def avg_loss(loss, n_words, **kwargs):
    return loss / n_words


def perplexity(loss, n_words, **kwargs):
    return math.exp(min(avg_loss(loss, n_words), 100))


def avg_support(support_size, n_words, **kwargs):
    return support_size / n_words


def gold_support_rate(n_supported, n_words, **kwargs):
    return 100 * n_supported / n_words


def precision(n_correct, support_size, **kwargs):
    # kinda silly
    return 100 * n_correct / support_size


class Statistics(object):

    def __init__(self, loss=0, n_words=0, n_correct=0, **kwargs):
        self._stats = kwargs
        self._stats['loss'] = loss
        self._stats['n_words'] = n_words
        self._stats['n_correct'] = n_correct
        self._stats['n_src_words'] = 0
        self.start_time = time.time()

        # Things to report every however many steps on the training set, and
        # the names to call them by
        """
        self._train_log_stats = [
            ('acc', accuracy), ('ppl', perplexity), ('xent', avg_loss)
        ]
        """
        self._train_log_stats = [
            ('acc', accuracy), ('loss', avg_loss),
            ('supp. size', avg_support), ('gold supp. rate', gold_support_rate)
        ]

        # Things to report for the validation set when it is time to
        # validate, and the names to call them by
        self._report_metrics = [
            ('loss', avg_loss), ('accuracy', accuracy),
            ('support size', avg_support),
            ('gold support rate', gold_support_rate)
        ]

    def add_src_lengths(self, src_lengths):
        self._stats['n_src_words'] += src_lengths

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, other, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            other: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not
        """
        for k, v in other.stats.items():
            if k in self._stats:
                if k != 'n_src_words' or update_n_src_words:
                    self._stats[k] += v
            else:
                # I think this is not intended behavior with update_n_src_words
                self._stats[k] = v

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Log statistics

        Args:
           step (int): current step
           num_steps (int): total steps
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_count = ["Step %2d/%5d" % (step, num_steps)]
        lr = ["lr: %7.5f" % learning_rate]
        metrics = [name + ": {:.2f}".format(m_func(**self._stats))
                   for name, m_func in self._train_log_stats]
        time_metrics = ["%3.0f/%3.0f tok/s; %6.0f sec" %
                        (self._stats['n_src_words'] / (t + 1e-5),
                         self._stats['n_words'] / (t + 1e-5),
                         time.time() - start)]

        train_log = "; ".join(chain(step_count, metrics, lr, time_metrics))

        logger.info(train_log)
        sys.stdout.flush()

    def report(self, dataset):
        template = dataset + ' {}: {:.3f}'  # not the same as %g
        for metric, m_func in self._report_metrics:
            logger.info(template.format(metric, m_func(**self._stats)))

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        # todo: make this flexible
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", avg_loss(**self._stats), step)
        writer.add_scalar(prefix + "/ppl", perplexity(**self._stats), step)
        writer.add_scalar(prefix + "/accuracy", accuracy(**self._stats), step)
        writer.add_scalar(prefix + "/tgtper", self._stats['n_words'] / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
