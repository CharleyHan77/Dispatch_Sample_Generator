#
#
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#
#
# # 输入一个数组，转换为一条单链表
# def createLinkedList(arr: 'List[int]') -> 'ListNode':
#     if arr is None or len(arr) == 0:
#         return None
#
#     head = ListNode(arr[0])
#     cur = head
#     for i in range(1, len(arr)):
#         cur.next = ListNode(arr[i])
#         cur = cur.next
#
#     return head
#
# arr = [100, 200, 300, 400, 500]
# head = createLinkedList(arr)
# #print(head.val)ta


class Node:
    def __init__(self, x: int):
        self.val = x
        self.next = None


class MyLinkedList:

    def __init__(self):
        self.head = Node(None)

    def get(self, index: int) -> int:
        p = self.head
        for _ in range(index + 1):
            p = p.next
            if not p.next:
                return -1
        return p.val

    def addAtHead(self, val: int) -> None:
        head = self.head
        new_node = Node(val)
        new_node.next = head
        head = new_node

    def addAtTail(self, val: int) -> None:
        p = self.head
        while p.next:
            p = p.next
        new_node = Node(val)
        p.next = new_node

    def addAtIndex(self, index: int, val: int) -> None:
        p = self.head
        for _ in range(index):
            p = p.next

        new_node = Node(val)
        new_node.next = p.next
        p.next = new_node

    def deleteAtIndex(self, index: int) -> None:
        p = self.head
        for _ in range(index):
            p = p.next

        to_delete = p.next
        p.next = to_delete.next
        to_delete = None

# Your MyLinkedList object will be instantiated and called as such:
obj = MyLinkedList()
param_1 = obj.addAtHead(3)g