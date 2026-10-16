class ComponentRegistry:
    def __init__(self):
        self._items = {}

    def register(self, name, obj):
        self._items[name] = obj

    def get(self, name):
        return self._items.get(name)

    def status(self):
        return list(self._items.keys())
