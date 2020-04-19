from math import log2
from random import randint
from collections import deque
import time

from tktree import DrawTreeByLink, DrawTreeByList

class NormalTree(object):
	MAXNODE = 10
	def __init__(self):
		self._root = TreeNode()
		self._last = None
		self._size = 0
		self.levelWidth, self.ndxOfLeftmost, self.ndxOfRightmost = {}, {}, {}
		self.lastDeletedIDs = []
		self.idDict = {}

	def __iter__(self):
		self.curNode = self._root
		return self

	def __next__(self):
		if self.curNode is not None:
			toReturn = self.curNode
			self.curNode = self.curNode.next
			return toReturn
		else:
			raise StopIteration

	def __str__(self):
		string = str()
		for x in self:
			string += x.content
			string += ','
		return string

	def _checkAndModifyID(self, ID):
		if ID in self.idDict.keys():
			ID = ID + self.size
			while ID in self.idDict.keys():
				ID += 1  
		return ID

	def _findNeighborNode(self, node):
		siblings = node.parent.children
		prevNode = nextNode = None
		if node.level not in self.levelWidth.keys():  # new level
			prevNode = self._last
		else:
			if len(siblings) > 1:     # neighbor is a sibling.
				left = [k for k in siblings.keys() if k < node.ndxInSib]
				right = [k for k in siblings.keys() if k > node.ndxInSib]
				if len(left) > 0:
					prevNode = siblings[max(left)]
				if len(right) > 0:
					nextNode = siblings[min(right)]
			else:     	# neighbor is a cousin.
				curNode = node.parent.prev
				while curNode is not None and len(curNode.children) == 0 and curNode.level == node.parent.level:
					curNode = curNode.prev
				if curNode.level == node.parent.level:
					prevNode = curNode.children[max(curNode.children.keys())]
				else:
					curNode = node.parent.next
					while curNode is not None and len(curNode.children) == 0:
						curNode = curNode.next
					if len(curNode.children) > 0:
						nextNode = curNode.children[min(curNode.children.keys())]
					elif curNode is None:
						prevNode = self._last
		if prevNode is None:
			prevNode = nextNode.prev
		elif nextNode is None:
			nextNode = prevNode.next
		return prevNode, nextNode

	def _findNdxInSib(self, siblings, ndxInSib = -1):
		'''ndxInSib = -1 means to append it after the last child.'''
		if len(siblings) != 0:
			if ndxInSib == -1:
				ndxInSib = max(siblings.keys()) +1
			elif ndxInSib in siblings.keys():  # to allocate position for the new bad kid
				k = ndxInSib + 1
				while k in siblings.keys():
					k += 1
				while k > ndxInSib:
					siblings[k] = siblings[k-1]
					k -= 1
		elif len(siblings) == 0 and ndxInSib == -1:
			ndxInSib = 0
		return ndxInSib
				
	def	_updateIDDict(self, **kw):
		''' 1. kw: newNode. 2. the ID identity is not checked here. It shall be checked before.'''
		if 'newNode' in kw:
			newNode = kw['newNode']
			id = newNode.nodeID
			self.idDict[id] = newNode
		else:
			self.idDict = {}
			curNode = self._root
			while curNode is not None:
				self.idDict[curNode.nodeID] = curNode
				curNode = curNode.next
			
	def _updateBFTLink(self, **kw):
		'''kw: updateAll, insertedNode'''
		if 'updateAll' in kw and kw['updateAll'] == True:
			qu = deque()
			curNode = self._root
			curNode.ndx = curNode.level = 1
			qu.append(self._root)
			while len(qu) != 0:
				children = dict(sorted(qu[0].children.items(),key = lambda x: x[0])).values()
				for ch in children:
					qu.append(ch)
					curNode.next = qu[-1]
					curNode.next.prev = curNode
					curNode.next.ndx = curNode.ndx + 1
					curNode = curNode.next
					curNode.level = curNode.parent.level + 1
				qu.popleft() 
			self._last = curNode
		elif 'insertedNode' in kw:
			newNode = kw['insertedNode']
			prevNode, nextNode = self._findNeighborNode(newNode)
			if nextNode is not None:  # not the last node
				prevNode.next = nextNode.prev = newNode
				newNode.prev, newNode.next= prevNode, nextNode
			elif prevNode == self._last:
				prevNode.next = newNode
				newNode.prev = prevNode
				self._last = newNode
			curNode = newNode
			while curNode is not None:
				curNode.ndx = curNode.prev.ndx + 1
				curNode.level = curNode.parent.level + 1
				curNode = curNode.next
			# self._updateTreeInfo(newNode)

	def _updateTreeInfo(self, fromNode):
		''' to update level, width, cousins, XY...info of the tree since the fromNode.'''
		self.height = self._last.level
		curNode = fromNode
		while curNode.prev is not None and curNode.prev.level == curNode.level:
			curNode = curNode.prev
		for level in range(curNode.level, self.height + 1):
			self.levelWidth[level] = 0
			while curNode is not None and curNode.level == level:
				self.levelWidth[level] += 1
				curNode = curNode.next
		self.width = max(self.levelWidth.values())

		curNode = fromNode    # can only be done after the above cycle finished. So cannot be combined in one cycle.
		self.levelWidth[0] = self.ndxOfLeftmost[0] = self.ndxOfRightmost[0] = 0 
		for l in range(curNode.level, self.height + 1):
			self.ndxOfRightmost[l] = self.ndxOfRightmost[l-1] + self.levelWidth[l]
			self.ndxOfLeftmost[l] = self.ndxOfRightmost[l] - self.levelWidth[l] + 1	

	def _updateCousinLinks(self):
		# to maintain the l/r cousins (which are of the same level)   
		curNode = self._root
		while curNode is not None:
			_ndx = curNode.ndx
			curNode.X = _ndx - self.ndxOfLeftmost[curNode.level] + 1
			curNode.lCou = curNode.prev
			curNode.rCou = curNode.next
			if _ndx == 1:
				curNode.lCou = curNode.rCou = curNode 
			elif curNode.X == 1:
				if len(curNode.prev.children) > 1:
					curNode.lCou = curNode.prev.children[len(curNode.prev.children)-1]  # left cousin: cycle to the rightmost 
					curNode.lCou.rCou = curNode
				else:
					curNode.lCou = None
			elif curNode.parent.X == self.levelWidth[curNode.level - 1] and curNode.ndxInSib == self.MAXNODE - 1:
				curNode.rCou = curNode.parent.next
				curNode.rCou.lCou =  curNode
				curNode.lCou.rCou = curNode
			else:
				curNode.lCou.rCou = curNode
			curNode = curNode.next

	def _updateDesendants(self, downFromNode):
		downFromNode.descendants = []
		for dNode in downFromNode.children.values():
			if len(dNode.children) > 0:
				self._updateDesendants(dNode)
				downFromNode.descendants.extend(dNode.descendants)
			downFromNode.descendants.append(dNode)

	def deleteDown(self, deleteID):
		downFromNode = self.idDict[deleteID]
		self._updateDesendants(downFromNode)
		allDeletedNodes = downFromNode.descendants + [downFromNode]
		self.lastDeletedIDs = []
		for x in allDeletedNodes:
			self.lastDeletedIDs.append(x.nodeID)
			del self.idDict[x.nodeID]
		if downFromNode is not self._root:
			x = downFromNode.ndxInSib
			del downFromNode.parent.children[x] 
			downFromNode.parent = None
		else: 
			del self
		self._updateBFTLink(updateAll = True)
		self._updateTreeInfo(self._root)
			
	def insert(self, parentID, insertID, content, **kw):
		'''1. ID shall be unique, otherwise it will be modified.
		2. kw: ndxInSib : the index of the node in its siblings. Default the last.'''
		checkedID = self._checkAndModifyID(insertID)
		newNode = TreeNode(checkedID, content, parent = self.idDict[parentID])
		self.idDict[checkedID] = newNode
		
		siblings = self.idDict[parentID].children
		if 'ndxInSib' in kw:
			ndxInSib = kw['ndxInSib']
			if ndxInSib in siblings.keys():  # to allocate position for the new bad kid
				k = ndxInSib + 1
				while k in siblings.keys():
					k += 1
				while k > ndxInSib:
					siblings[k] = siblings[k-1]
					k -= 1
		elif len(siblings) > 0:
			ndxInSib = max(siblings.keys()) +1
		else:
			ndxInSib = 0
		siblings[ndxInSib] = newNode
		self._updateBFTLink(insertedNode = newNode)

	def moveToTree(self, subTreeRootID, toTree, toParentID, ndxInSib = -1):
		'''move subtree(with root ID) of tree(self) to toTree.toParentID.children[ndxInSib](default the last position).'''
		subTreeRootNode = self.idDict[subTreeRootID]
		self._updateDesendants(subTreeRootNode)
		self.deleteDown(subTreeRootID)    # only delete the links to original tree. Nodes are still availible.

		toParentNode = toTree.idDict[toParentID]
		ndxInSib = toTree._findNdxInSib(toParentNode.children)
		subTreeRootNode.parent = toParentNode
		toParentNode.children[ndxInSib] = subTreeRootNode

		levelOffset = toParentNode.level + 1 - subTreeRootNode.level
		allMoved = subTreeRootNode.descendants + [subTreeRootNode]
		allMoved = sorted(allMoved, key = lambda x: x.nodeID, reverse = True)
		for node in allMoved:
			node.level += levelOffset
			node.nodeID = toTree._checkAndModifyID(node.nodeID) 
			toTree.idDict[node.nodeID] = node
		
		toTree._updateBFTLink(updateAll = True)
		self._updateBFTLink(updateAll = True)
		toTree._updateTreeInfo(toTree._root)
		self._updateTreeInfo(self._root)

	@property
	def size(self):
		return self._last.ndx

class BinSearchTree(NormalTree):
	def init(self):
		super().__init__()

	def _findParentAndIndex(self, value, forSearch = False):
		curNode = self._root
		while True:
			if value < curNode.content:
				if 0 in curNode.children:
					curNode = curNode.children[0]
				else: 
					n = 0
					return curNode, n
			elif value > curNode.content:
				if 1 in curNode.children:
					curNode = curNode.children[1]
				else: 
					n = 1
					return curNode, n
			elif value == curNode.content:
				if forSearch:
					return curNode, 'found'
				else: 
					if 0 in curNode.children:
						curNode = curNode.children[0]
					else: 
						n = 0
						return curNode, n

	def buildFromSeq(self, seq):
		assert len(seq) > 0, 'the sequence shall not be empty.'
		self._root = TreeNode(0, seq[0])
		for i in range(1, len(seq)):
			pa, n = self._findParentAndIndex(seq[i])
			insertNode = TreeNode(i, seq[i], parent = pa)
			pa.children[n] = insertNode
		self._updateBFTLink(updateAll = True)
		self._updateIDDict()
		self._updateTreeInfo(self._root)

	def insertNode(self, value):
		pa, n = self._findParentAndIndex(value)
		insertNode = TreeNode(self.size + 1, value, parent = pa)
		pa.children[n] = insertNode
		self._updateBFTLink(insertedNode = insertNode)
		self._updateIDDict(newNode = insertNode)
		self._updateTreeInfo(insertNode)
	
	def search(self, value):
		node, result = self._findParentAndIndex(value, True)
		if result == 'found':
			return node
	
	def _findClosest(self, node):
		leftClosest, rightClosest = None, None
		if node.children != {}:
			if 0 in node.children:
				leftClosest = node.children[0]
				while 1 in leftClosest.children:
					leftClosest = leftClosest.children[1]
			if 1 in node.children:
				rightClosest = node.children[1]
				while 0 in rightClosest.children:
					rightClosest = rightClosest.children[0]
		return leftClosest, rightClosest

	def deleteValue(self, value):
		node = self.search(value)
		child = None
		if node is not None:
			l, r = self._findClosest(node)
			if l is None and r is None:
				del node.parent.children[node.ndxInSib]
				del node
			elif l is None:
				toCopy = r
				if 1 in r.children:
					child = r.children[1]
			elif r is None:
				toCopy = l
				if 0 in l.children:
					child = l.children[0]
			else:
				if node.content - l.content <= r.content - node.content:
					toCopy = l
					if 0 in l.children:
						child = l.children[0]
				else: 
					toCopy = r
					if 1 in r.children:
						child = r.children[1]
		
			node.content = toCopy.content
			if child is not None:
				child.parent = toCopy.parent
				toCopy.parent.children[toCopy.ndxInSib] = child
			del toCopy 
			self._updateBFTLink(updateAll = True)
			self._updateIDDict()
			self._updateTreeInfo(self._root)
			
class CompleteBinTreeByLink(NormalTree):
	MAXNODE = 2
	def __init__(self):
		super().__init__()
			
	def buildFromSeq(self, seq):
		''' to build a tree from an non-empty sequence.'''
		assert len(seq) != 0, 'the seq cannot be empty.'
		self._root = self._last = TreeNode(1, seq[0])
		self._size = len(seq)
		qu = deque()
		qu.append(self._root)
		n = 1
		while n < len(seq):
			content = seq[n]
			if 0 not in qu[0].children.keys():   # child No.0 = left child
				qu[0].children[0] = self._last.next = TreeNode(n + 1, content, parent = qu[0])
				self._last.next.prev = self._last
				self._last = self._last.next
				qu.append(self._last)
				n += 1
			elif 1 not in qu[0].children.keys():   # child No.1 = right child
				qu[0].children[1] = self._last.next = TreeNode(n + 1, content, parent = qu[0])
				self._last.next.prev = self._last
				self._last = self._last.next
				qu.append(self._last)
				n += 1
			else: 
				qu.popleft()
		self._updateIDDict()
		self._updateTreeInfo(self._root)

	def append(self, *allContent):
		for content in allContent:
			self.idDict[self._size+1] = newNode = TreeNode(self._size + 1, content)
			if self._isPerfectBTree():
				curNode = self._last.rCou
				curNode.children[0] = newNode
			elif self._last.ndxInSib == self.MAXNODE - 1: 
				curNode = self._last.parent.rCou
				curNode.children[0] = newNode
			else:
				curNode = self._last.parent
				curNode.children[len(curNode.children)] = newNode
			newNode.parent = curNode
			self._last.next = newNode
			self._last.next.prev = self._last
			self._last = newNode
			self._size += 1
			self._updateTreeInfo(self._last)

	def deleteAndMove(self, **kw):
		''' kw: ID, node, fromNode, toNode, fromID, toID '''
		''' by default, delete the last node.  '''
		fromNode = toNode = self._last
		if 'node' in kw:
			fromNode = toNode = kw['node']
		elif 'ID' in kw:
			fromNode = toNode = self.idDict[kw['ID']]
		if 'fromNode' in kw and 'toNode' in kw:
			fromNode, toNode = kw['fromNode'], kw['toNode']
		elif 'fromID' in kw and 'toID' in kw:
			fromNode, toNode = self.idDict[kw['fromID']], self.idDict[kw['toID']]
		assert fromNode is not None and toNode is not None, 'the node has already been deleted.'

		prevOfDeleted = fromNode.prev
		if prevOfDeleted is None:
			prevOfDeleted = self._root

		loop = True
		while loop:
			if toNode.next is not None:
				fromNode.nodeID = toNode.next.nodeID
				fromNode.content = toNode.next.content
				fromNode, toNode = fromNode.next, toNode.next
			else:
				self._last = fromNode.prev
				self._last.next = None
				loop = False

		self._updateTreeInfo(prevOfDeleted)

	def _isPerfectBTree(self):
		return (self._last.ndx & (self._last.ndx + 1) == 0) 

class ExpressionTree(NormalTree):
	TOKENS = ('+','-','*','/')
	def __init__(self):
		super().__init__()

	def _moveDown(self, node, left = True, right = False):
		newNode = TreeNode()
		if node == self._root:
			self._root = newNode
		else:
			node.parent.children[node.ndxInSib] = newNode
			newNode.parent = node.parent
		node.parent = newNode
		if left == True and right == False:
			newNode.children[0] = node
		elif right == True and left == False:
			newNode.children[1] = node
		return newNode
	
	def _addRightSib(self, leftNode, rightNode = None):
		if rightNode is None:
			rightNode = TreeNode()
		leftNode.parent.children[leftNode.ndxInSib + 1] = rightNode
		rightNode.parent = leftNode.parent
		return rightNode
	
	def _transformToPostfixExpr(self, expr):
		stack = deque()
		pfExpr = []
		priorty = {'+':1,'-':1,'*':2,'/':2,'(':0,')':0}
		n = 0 
		while n < len(expr):
			_type, ele, n = self._getNext(expr, n)
			if _type == 'digit' or _type == 'var':
				pfExpr.append(ele)
			elif _type == 'token':
				if len(stack) == 0:
					stack.append(ele)
				elif priorty[ele] > priorty[stack[-1]]:
					stack.append(ele)
				else:
					pfExpr.append(stack.pop())
					stack.append(ele)
			elif _type == 'paren':
				if ele == '(':
					stack.append(ele)
				elif ele == ')':
					while stack[-1] != '(':
						pfExpr.append(stack.pop())
					stack.pop()
		while len(stack) != 0:
			pfExpr.append(stack.pop())
		return pfExpr

	def _getNext(self, expr, n):
		while expr[n].isspace():
			n += 1
		if expr[n] in self.TOKENS:
			return 'token', expr[n], n + 1
		elif expr[n].isdecimal():
			n2 = n + 1
			while n2 < len(expr) and expr[n2].isdecimal():
				n2 += 1
			return 'digit', int(expr[n:n2]), n2
		elif expr[n] in ('(',')'):
			return 'paren', expr[n], n + 1
		elif n2 < len(expr) and expr[n].isalpha():
			n2 = n + 1
			while expr[n2].isalpha():
				n2 += 1
			return 'var', str(expr[n:n2]), n + 1
			
	def buildFromExpr(self, expr):
		assert len(expr) != 0, 'expr error.'
		pfExpr = self._transformToPostfixExpr(expr)
		self._root = TreeNode(1, pfExpr[0])
		curNode = self._root
		for n in range(len(pfExpr)-1):
			if type(pfExpr[n+1]) == int:
				if curNode.content is None:
					curNode.content = pfExpr[n]
				curNode = self._moveDown(curNode)
				curNode = self._addRightSib(curNode.children[0], \
											TreeNode(-1, pfExpr[n+1], parent = curNode))
			elif pfExpr[n+1] in self.TOKENS:
				if curNode.parent is not None:
					curNode = curNode.parent
					curNode.content = pfExpr[n+1]
				else:
					curNode = self._moveDown(curNode, False, True)
		self._updateBFTLink(updateAll = True)
		self._updateIDDict()
		self._updateTreeInfo(self._root)

	def evaluate(self, fromNode):
		elements = []
		if len(fromNode.children) == 0:
			return fromNode.content
		else:
			for ch in fromNode.children.values():
				elements.append(self.evaluate(ch))
			elements.append(fromNode.content)
			return self._compute(elements)

	def _compute(self, elements):
		operator = elements[-1]
		if operator == '+': return elements[0] + elements[1]
		if operator == '-': return elements[0] - elements[1]
		if operator == '*': return elements[0] * elements[1]
		if operator == '/': return elements[0] / elements[1]

class CompleteBinTreeByList(list):
	def __init__(self):
		super().__init__()

	@property
	def _root(self):
		if len(self) > 0:
			return self[0]
		else: return None

	@property
	def _last(self):
		if len(self) > 0:
			return self[-1]
		else: return None

	@property
	def height(self):
		return int(log2(len(self)))+1

	@property
	def width(self):
		l1 = pow(2, self.height - 2)
		l2 = len(self) - pow(2, self.height - 1) + 1
		return max(l1, l2) 

	@property
	def size(self):
		return len(self)
	
	def buildFromRandom(self, _size):
		for i in range(_size):
			self.append(TreeNodeInList(self, i, randint(0,20)))
	
	def buildFromSeq(self, seq):
		for i in range(len(seq)):
			self.append(TreeNodeInList(self, i, seq[i]))

class HeapTree(CompleteBinTreeByList):
	def __init__(self):
		super().__init__()

	def minSort(self, fromNode):
		if fromNode.children != {}:
			self._recMinSort(fromNode)
			for ch in fromNode.children.values():
				self._recMinSort(ch)
	
	def _recMinSort(self, node):
		if node.children == {}:
			return node
		else:
			lChild = self._recMinSort(node.children[0])
			smaller = lChild
			if 1 in node.children:
				rChild = self._recMinSort(node.children[1])
				if rChild.content < lChild.content :
					smaller = rChild 
			if smaller.content < node.content:
				node.content, smaller.content = smaller.content, node.content
			return node

	def heapAppend(self, content):
		self.append(TreeNodeInList(self, len(self), content))
		curNode = self[-1]
		while curNode.content < curNode.parent.content and curNode != self[0]:
			curNode.content, curNode.parent.content = \
				curNode.parent.content, curNode.content
			curNode = curNode.parent

	def headExtract(self):
		tmp = self[0].content
		if self.size > 1:
			curNode = self[0] = self.pop()
			loop = True
			while loop:
				if curNode.children != {}:
					smaller = curNode.children[0]
					if 1 in curNode.children and curNode.children[1].content < smaller.content:
						smaller = curNode.children[1]
					if curNode.content > smaller.content:
						curNode.content, smaller.content = smaller.content, curNode.content
						curNode = smaller
					else: loop = False
				else: loop = False
		else: 
			self.pop()
		return tmp
			
class TreeNode(object):
	'''nodeID, content: -1 and None means to be updated later. kw: parent'''
	def __init__(self, nodeID = -1, content = None, **kw):
		self.nodeID = nodeID
		self.content = content
		self.children = {}
		self._ndxInSib = None
		self.lCou = None	# for traversal in layer, in cycle mode.
		self.rCou = None	
		self.next = None  # traversal the treenodes in the breath first mode.
		self.prev = None
		self.ndx = nodeID   #equal to id when initiate. ID is fixed and ndx if mutable.For calc position in the drawing.
		self.X = self._level = None
		self.drawXY = ()    # for drawing info.
		self.drawWidth = None  # for drawing info.
		if 'parent' in kw:
			self.parent = kw['parent']
			self._level = self.parent._level + 1
		else: 
			self.parent = None
			self._level = 1
		self.descendants = []

	def __str__(self):
		if type(self.content) == int:
			return ('nodeID, content, ndx: %d, %d, %d' % (self.nodeID, self.content, self.ndx))
		if type(self.content) == str:
			return ('nodeID, content, ndx: %d, %s, %d' % (self.nodeID, self.content, self.ndx))

	@property
	def ndxInSib(self):
		_siblings = self.parent.children
		for k in _siblings.keys():
			if _siblings[k] == self:
				self._ndxInSib = k
		return self._ndxInSib

	@ndxInSib.setter
	def ndxInSib(self, value):
		self._ndxInSib = value

	@property
	def siblings(self):
		if self.parent is not None:
			return self.parent.children
	
	@property
	def level(self):
		if self._level is not None:
			return self._level
		else:
			return self.parent.level + 1

	@level.setter
	def level(self, value):
		self._level = value

class TreeNodeInList(object):
	def __init__(self, inTree, nodeID = -1, content = None, **kw):
		# super().__init__(nodeID = -1, content = None, **kw)
		self.inTree = inTree
		self.nodeID = nodeID
		self.content = content
		self.drawWidth = None  # for drawing info.
		self.descendants = []

	@property
	def ndx(self):
		return self.inTree.index(self)
	
	@property
	def level(self):
		return int(log2(self.ndx+1))+1

	@property
	def children(self):
		l, r = (self.ndx + 1) * 2 - 1, (self.ndx + 1) * 2
		if r < len(self.inTree): return {0:self.inTree[l], 1:self.inTree[r]}
		elif l < len(self.inTree): return {0:self.inTree[l]} 
		else: return {}

	@property
	def parent(self):
		n = (self.ndx+1)//2-1
		if n < 0: return None
		else:
			return self.inTree[n]

	@property
	def ndxInSib(self):
		if self.ndx + 1 == (self.parent.ndx + 1) * 2:
			return 0
		elif self.ndx == (self.parent.ndx + 1) * 2:
			return 1

	@property
	def siblings(self):
		if self.ndx + 1 == self.parent.ndx * 2 and self.ndx + 1 < len(self.inTree):
			return self.inTree(self.ndx + 1)
		elif self.ndx == self.parent.ndx * 2 and self.ndx - 1 >= 0:
			return self.inTree(self.ndx - 1)
		else: return None

def testBinTree():
	tree1 = CompleteBinTreeByLink()
	# seq = list(chr(x+65) for x in range(0, 26))
	seq = list(x for x in range(1, 16))
	tree1.buildFromSeq(seq)
	draw1 = DrawTreeByLink(tree1)
	tree1.append(*(16,17))
	draw1.updateDrawing('append')
	tree1.deleteAndMove()
	draw1.updateDrawing('delete')
	tree1.deleteAndMove(ID = 3)
	draw1.updateDrawing('delete', ID = 3)
	tree1.deleteAndMove(fromID = 6, toID = 14)
	draw1.updateDrawing('delete', fromID = 6, toID = 14)

def testNormalTree():
	tree1 = CompleteBinTreeByLink()
	seq1 = list(x for x in range(1, 16))
	tree1.buildFromSeq(seq1)
	draw1 = DrawTreeByLink(tree1)

	tree2 = CompleteBinTreeByLink()
	seq2 = list(chr(x+65) for x in range(0, 10))
	tree2.buildFromSeq(seq2)
	draw2 = DrawTreeByLink(tree2)	

	# tree1.insert(13, 30, 'x')
	# draw1.updateDrawing('insert')
	# tree2.insert(6, 31, 'y', ndxInSib = 0)
	# draw2.updateDrawing('insert', insertID = 31)

	tree1.moveToTree(6, tree2, 8)
	draw1.updateDrawing('redraw')
	draw2.updateDrawing('redraw')

	# tree1.insert(11, 32, 'z', ndxInSib = 0)
	# draw1.updateDrawing('insert', insertID = 32)
	# tree1.deleteDown(deleteID = 5)
	# draw1.updateDrawing('deleteDown', lastDeletedIDs = tree1.lastDeletedIDs)

def testExpTree():
	tree = ExpressionTree()
	expr = '((1-2-4/5*(1/3))+(2+4+3*5)*5)/7'
	# ((1+2-4/5*(1/3))+(2+4+3*5)*5)/7
	# 1,2,+,4,5,/,1,3,/,*,-,2,4,+,3,5,*,+,5,*,+,7,/
	# expr = '1+2*3'
	tree.buildFromExpr(expr)
	draw = DrawTreeByLink(tree)
	draw.updateDrawing('redraw')
	value = tree.evaluate(tree._root)
	print(value)

def testHeapTree():
	htree = HeapTree()
	# htree.buildFromRandom(10)
	seq = [3, 17, 7, 12, 19, 5, 15, 2, 18, 6]
	htree.buildFromSeq(seq)
	htree.minSort(htree._root)
	draw = DrawTreeByList(htree)
	htree.heapAppend(4)
	draw.updateDrawing('redraw')
	sortedSeq = []
	while htree.size != 0:
		sortedSeq.append(htree.headExtract())
	# draw.updateDrawing('redraw')
	print(sortedSeq)

def testBSTree():
	bsTree = BinSearchTree()
	seq = [2, 15, 6, 19, 18, 9, 8, 10, 17, 1, 6, 0, 15, 16, 12]
	bsTree.buildFromSeq(seq)
	draw1 = DrawTreeByLink(bsTree)
	# bsTree.insertNode(14)
	# draw1.updateDrawing('redraw')
	# input('search: 8')
	# result = bsTree.search(8)
	# print(result)
	# input('search: 9')
	# result = bsTree.search(9)
	# print(result)
	# input('search: 30')
	# result = bsTree.search(30)
	# print(result)
	bsTree.deleteValue(9)
	draw1.updateDrawing('redraw')
	bsTree.deleteValue(19)
	draw1.updateDrawing('redraw')



# testBinTree()
# testNormalTree()
# testExpTree()
# testHeapTree()
testBSTree()


