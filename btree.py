class BTreeNode:
    def __init__(self, leaf=True):
        self.leaf = leaf
        self.keys = []
        self.values = []
        self.children = []


class BTree:
    def __init__(self, t=2):
        self.root = BTreeNode(True)
        self.t = t  # Minimum degree (defines bounds on keys: t-1 to 2t-1)

    def search(self, key, node=None):
        if node is None:
            node = self.root

        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i]

        if node.leaf:
            return None

        return self.search(key, node.children[i])

    def insert(self, key, value):
        # First check if the key already exists and update it in-place
        exists = self._update_if_exists(self.root, key, value)
        if exists:
            return

        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            temp = BTreeNode(False)
            self.root = temp
            temp.children.insert(0, root)
            self.split_child(temp, 0, root)
            self.insert_non_full(temp, key, value)
        else:
            self.insert_non_full(root, key, value)

    def _update_if_exists(self, node, key, value):
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            node.values[i] = value
            return True

        if node.leaf:
            return False

        return self._update_if_exists(node.children[i], key, value)

    def split_child(self, x, i, y):
        t = self.t
        z = BTreeNode(y.leaf)
        x.children.insert(i + 1, z)
        x.keys.insert(i, y.keys[t - 1])
        x.values.insert(i, y.values[t - 1])

        z.keys = y.keys[t : (2 * t) - 1]
        z.values = y.values[t : (2 * t) - 1]
        y.keys = y.keys[0 : t - 1]
        y.values = y.values[0 : t - 1]

        if not y.leaf:
            z.children = y.children[t : 2 * t]
            y.children = y.children[0 : t]

    def insert_non_full(self, x, key, value):
        i = len(x.keys) - 1
        if x.leaf:
            while i >= 0 and key < x.keys[i]:
                i -= 1
            i += 1
            x.keys.insert(i, key)
            x.values.insert(i, value)
        else:
            while i >= 0 and key < x.keys[i]:
                i -= 1
            i += 1
            if len(x.children[i].keys) == (2 * self.t) - 1:
                self.split_child(x, i, x.children[i])
                if key > x.keys[i]:
                    i += 1
            self.insert_non_full(x.children[i], key, value)

    def inorder(self, node=None):
        if node is None:
            node = self.root
        result = []
        self._inorder_traversal(node, result)
        return result

    def _inorder_traversal(self, node, result):
        i = 0
        for i in range(len(node.keys)):
            if not node.leaf:
                self._inorder_traversal(node.children[i], result)
            result.append((node.keys[i], node.values[i]))
        if not node.leaf:
            self._inorder_traversal(node.children[i + 1], result)
