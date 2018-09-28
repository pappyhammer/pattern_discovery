
# serve to represent the a sequence of spikes based on probability
# from one neuron can have many follower up to n_order.
# n_order == 1, then only the highest probabilty follower is on the child
class Tree(object):
    def __init__(self, neuron, n_order, father, prob=1, acc=0):
        # key are int, 0 representing the highest probability in the Tree
        self.child = dict()
        self.neuron = neuron
        self.n_order = n_order
        self.father = father
        # unique ID for each tree among members of the same root tree, list on neuron in the sequence till the gieven
        # neuron
        if father is None:
            self.id = f'{neuron}'
        else:
            self.id = father.id + '_' + f'{neuron}'
        # parents is a dict containing as a key an int representing the neurons
        # that are part of the sequence.
        self.parents = {}
        if father is not None:
            self.parents = dict(father.parents)
            self.parents[father.neuron] = 1
        self.own_prob = prob
        self.acc = acc
        if father is None:
            self.total_prob = prob
        else:
            self.total_prob = prob * father.total_prob

    def disinherit(self):
        """
        Remove this tree from his father
        :return:
        """
        if self.father is not None:
            del(self.father.child[self.id])
            # if a tree loose all his children, then it is disinherited as well
            if len(self.father.child) == 0:
                self.father.disinherit()

    def disinherit_2nd_order(self):
        """
                Remove this tree from his father and grandfather
                :return:
                """
        if self.father is not None:
            if self.id in self.father.child:
                del (self.father.child[self.id])
                if self.father.father is not None:
                    if self.father.id in self.father.father.child:
                        del (self.father.father.child[self.father.id])
                    # if a tree loose all his children, then it is disinherited as well
                    if len(self.father.child) == 0:
                        self.father.disinherit_2nd_order()
                    if len(self.father.father.child) == 0:
                        self.father.father.disinherit_2nd_order()

    def max_seq_by_total_prob(self):
        seq = [self.neuron]
        if len(self.child) > 0:
            max_child = None
            for k, child_v in self.child.items():
                if max_child is None:
                    max_child = k
                else:
                    if self.child[k].total_prob < child_v.total_prob:
                        max_child = k
            seq.extend(self.child[max_child].max_seq_by_total_prob())
        return seq

    def add_child(self, child):
        self.child[child.id] = child

    # not used, delete the child from root than will give birth to self
    def cut_from_the_root(self, identifier=None):
        if identifier is None:
            if self.father is None:
                # shouldn't happen
                return
            self.father.cut_from_the_root(identifier=self.id)
        if self.father is None:
            # in case it would already have been deleted
            if identifier in self.child:
                del(self.child[identifier])
        else:
            self.father.cut_from_the_root(identifier=self.id)

    def __str__(self):
        result = f"Tree: neuron {self.neuron}\n"
        result += '\n'
        result += f"{len(self.child)} children:\n"
        for v in self.child:
            result += str(v)
        return result

    def __lt__(self, other):
        """
        inferior self < other
        :param other:
        :return:
        """
        return self.acc < other.acc

    def __le__(self, other):
        """
        Lower self <= other
        :param other:
        :return:
        """
        return self.acc <= other.acc

    def __eq__(self, other):
        """
        Equal self == other
        :param other:
        :return:
        """
        return self.acc == other.acc

    def __ne__(self, other):
        """
        non equal self != other
        :param other:
        :return:
        """
        return self.acc != other.acc

    def __gt__(self, other):
        """
        Greater self > other
        :param other:
        :return:
        """
        return self.acc > other.acc

    def __ge__(self, other):
        """
        Greater self >= other
        :param other:
        :return:
        """
        return self.acc >= other.acc

    def is_in_the_tree(self, n):
        if self.neuron == n:
            return True
        if self.father is None:
            return False
        return self.father.is_in_the_tree(n)

    def get_seq_lists(self):
        # print(f'get_seq_lists neuron {self.neuron}')
        if len(self.child) == 0:
            # print(f'get_seq_lists neuron2 {self.neuron}')
            return [[self.neuron]]
        result = []
        # print(f'get_seq_lists result1: {result}')
        for tree_child in self.child.values():
            child_lists = tree_child.get_seq_lists()
            # print(f'get_seq_lists child_lists {child_lists}')
            for child_l in child_lists:
                # print(f'get_seq_lists child_l {child_l}')
                tmp_list = [self.neuron]
                tmp_list.extend(child_l)
                result.append(tmp_list)
                # print(f'get_seq_lists result2: {result}')
        # print(f'get_seq_lists result: {result}')
        return result

