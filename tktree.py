import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as mbox
import time

import win32api,win32con

class BiTreeForTreeviewTTK(object):
	def __init__(self, items):
		self.master = tk.Tk()
		self.items = items

		tree = ttk.Treeview(height = len(self.items), show = 'tree')
		tree.column('#0', width = 400)
		assert len(self.items) > 0, 'the tree shall not be empty.'
		tree.insert('', 0, items[0], text = items[0])
		for i in range(1, len(items)):
			tree.insert(items[(i-1)//2], i, items[i], text = items[i])
			tree.see(items[i])
		tree.pack()

	def mainloop(self):
		self.master.mainloop()

class DrawTreeByLink(object):
	def __init__(self, tree):
		self.tree = tree
		self._initTreeSize = self.tree.width, self.tree.height
		self.root = tk.Tk()

		self.step = tk.IntVar(master = self.root, value = 0)

		self.fmCanvas = tk.Frame(self.root)
		self.fmVerBar = tk.Frame(self.fmCanvas)
		self.fmHorBar = tk.Frame(self.fmCanvas)
		self.fmForButtons = tk.Frame(self.root)

		self.fmForButtons.pack(side = 'top', fill = 'both', expand = True)
		self.fmCanvas.pack(side = 'top', fill = 'both', expand = True)
		self.fmVerBar.pack(side = 'right', fill = 'both', expand = False)
		self.fmHorBar.pack(side = 'bottom', fill = 'both', expand = False)

		self.bt1 = tk.Button(self.fmForButtons, text = '   next   ', command = lambda: self.step.set(self.step.get()+1))
		self.bt2 = tk.Button(self.fmForButtons, text = 'button-2')
		self.bt1.pack(side = 'left')
		self.bt2.pack(side = 'left')
		
		cx = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
		cy = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
		self.zoomX, self.zoomY = self.tree.width/2 * 1440/cx, self.tree.height/2 * 720/cy
		if self.tree.height > 5:	self.zoomY *= 0.7
		if self.tree.width > 8:		self.zoomX *= 0.7
		self.root.geometry('%dx%d+%d+%d' %(cx*self.zoomX*0.2, cy*self.zoomY*0.3, 100, 100))

		self._drawCanvas()
		self._drawNodesAndLines(fromNode = self.tree._root, toNode = self.tree._last)
		# self.root.update()
		self.bt1.waitvar(self.step)
		# mbox.showinfo('', 'inited.')
		# self.root.mainloop()

	def _drawCanvas(self):
	
		self.baseSize = 40 
		self.canvasWidth = self.baseSize * self.tree.width * self.zoomX * 1.5
		self.canvasHeight = self.baseSize * (self.tree.height + 1)* self.zoomY
		self.offset = self.baseSize // 2
		self.cv = tk.Canvas(self.fmCanvas, width = self.canvasWidth, height = self.canvasHeight, borderwidth = 3, relief = 'ridge')
		self.cv.pack(side = 'left', fill = 'both', expand = True)

		self.verBar = tk.Scrollbar(self.fmVerBar, width = 20)
		self.verBar.pack(side = 'right', fill = 'both', expand = False)
		self.horBar = tk.Scrollbar(self.fmHorBar, orient = 'horizontal', width = 20)
		self.horBar.pack(side = 'bottom', fill = 'both', expand = False)
		self.horBar.config(command = self.cv.xview)
		self.verBar.config(command = self.cv.yview)
		self.cv.config(yscrollcommand = self.verBar.set, xscrollcommand = self.horBar.set, 
					scrollregion = (0, 0, self.canvasWidth, self.canvasHeight))

	def _drawNodesAndLines(self, **kw):
		drawList = []
		fromNode = toNode = None
		if 'fromNode' in kw and 'toNode' in kw:
			fromNode, toNode = kw['fromNode'], kw['toNode']
			curNode = fromNode
			while curNode != toNode.next:
				drawList.append(curNode) 
				curNode = curNode.next
		elif 'fromID' in kw and 'toID' in kw:
			fromNode, toNode = self.tree.idDict[kw['fromID']], self.tree.idDict[kw['toID']]
			curNode = fromNode
			while curNode != toNode.next:
				drawList.append(curNode) 
				curNode = curNode.next
		elif 'downFromNode' in kw:
			self.tree._updateDesendants(kw['downFromNode'])
			drawList = kw['downFromNode'].descendants
		else:
			mbox.showerror('', 'nothing drawed.')
			return

		# draw nodes:
		n = 0
		while n < len(drawList):
			curNode = drawList[n]
			if curNode.level == 1:
				curNode.drawXY = 0.5 * self.canvasWidth, self.offset
				curNode.drawWidth = self.canvasWidth
			else:
				tmp = max(1, max(curNode.parent.children.keys())) 
				if curNode.drawWidth is None:
					curNode.drawWidth = curNode.parent.drawWidth / (1 + tmp)
				curNode.drawXY = curNode.parent.drawXY[0] - 1/2 * curNode.parent.drawWidth + \
								(0.5 + curNode.ndxInSib) * curNode.drawWidth, \
								(curNode.level-1)*self.baseSize*self.zoomY + self.offset			
				if curNode.drawXY[1] == curNode.prev.drawXY[1] and \
						curNode.drawXY[0] - curNode.prev.drawXY[0] < self.baseSize * 1.1:
					curNode.drawXY = curNode.drawXY[0] + self.baseSize, curNode.drawXY[1]
					self.canvasWidth += self.baseSize
					if curNode.ndxInSib == 0: 
						pmove = self.baseSize
					else: pmove = self.baseSize/2
					while curNode.parent != self.tree._root:
						for sib in curNode.parent.siblings.values():
							if sib.ndxInSib >= curNode.parent.ndxInSib:
								sib.drawXY = sib.drawXY[0] + pmove, sib.drawXY[1]
						curNode = curNode.parent
			n += 1

		for curNode in drawList:
			if len(curNode.children) > 1:
				x = 0
				for ch in curNode.children.values():
					x += ch.drawXY[0]
				curNode.drawXY = x/len(curNode.children), curNode.drawXY[1]
			elif 0 in curNode.children:
				curNode.drawXY = curNode.children[0].drawXY[0] + curNode.children[0].drawWidth, curNode.drawXY[1]
			elif 1 in curNode.children:
				curNode.drawXY = curNode.children[1].drawXY[0] - curNode.children[1].drawWidth, curNode.drawXY[1]

		for curNode in drawList:
			x1, y1 = curNode.drawXY
			x2, y2 = x1 + self.baseSize, y1 + self.baseSize
			tmpTag = ('nodes', 'node'+str(curNode.nodeID))
			self.cv.create_oval(x1, y1, x2, y2, fill = 'white', tag = tmpTag)
			self.cv.create_text(x1 + self.offset, y1 + self.offset, text = curNode.content, tag = tmpTag)

		#draw lines:
		for curNode in drawList:
			for ch in curNode.children.values():
				x1, y1, x2, y2 = curNode.drawXY[0] + self.offset, curNode.drawXY[1] + self.offset,  \
					ch.drawXY[0] + self.offset, ch.drawXY[1] + self.offset
				tmpTag = ('lines', str(curNode.nodeID) + '->', '->'+ str(ch.nodeID))
				self.cv.create_line(x1, y1, x2, y2, width = 3, fill = 'red', tag = tmpTag)
			if curNode.parent is not None:   # to draw for newly added nodes.
				tmpTag = ('lines', str(curNode.parent.nodeID) + '->', '->' + str(curNode.nodeID))
				if self.cv.find('withtag', tmpTag) == ():
					parentXY = curNode.parent.drawXY[0] + self.offset, curNode.parent.drawXY[1] + self.offset
					x1, y1, x2, y2 =  parentXY[0], parentXY[1], 	\
						curNode.drawXY[0] + self.offset, curNode.drawXY[1] + self.offset
					self.cv.create_line(x1, y1, x2, y2, width = 3, fill = 'red', tag = tmpTag)					

		#draw cousin lines:
		for curNode in drawList:
			if curNode.lCou is not None:
				x1, y1, x2, y2 = curNode.drawXY[0] + self.offset, curNode.drawXY[1] + self.offset, \
						curNode.lCou.drawXY[0] + self.offset, curNode.lCou.drawXY[1] + self.offset
				tmpTag = ('lines', str(curNode.nodeID) + '->', '->'+ str(curNode.lCou.nodeID))
				self.cv.create_line(x1, y1, x2, y2, width = 2, fill = 'green', tag = tmpTag)
			if curNode.rCou is not None:
				x1, y1, x2, y2 = curNode.drawXY[0] + self.offset, curNode.drawXY[1] + self.offset, \
						curNode.rCou.drawXY[0] + self.offset, curNode.rCou.drawXY[1] + self.offset
				tmpTag = ('lines', str(curNode.nodeID) + '->', '->'+ str(curNode.rCou.nodeID))
				self.cv.create_line(x1, y1, x2, y2, width = 2, fill = 'green', tag = tmpTag)

		self.cv.lift('nodes')
		self._lastDrawNode = drawList[-1]

	def _deleteNodeAndLines(self, **kw):
		'''kw: deleteAll, downFromNode, fromID/toID, lastDeletedIDs '''
		if 'deleteAll' in kw:
			self.cv.destroy()
			self._drawCanvas()
		elif 'downFromNode' in kw:
			self.tree._updateDesendants(kw['downFromNode'])
			for node in kw['downFromNode'].descendants:
				tmpTag = 'node'+str(node.nodeID), str(node.nodeID)+'->', '->'+str(node.nodeID)
				self.cv.delete(*tmpTag)
		elif 'fromID' in kw and 'toID' in kw:
			fromID, toID = kw['fromID'], kw['toID']
			endID = max(self._lastDrawNode.nodeID, toID)
			for id in range(fromID, endID + 1):
				tmpTag = 'node'+str(id), str(id)+'->', '->'+str(id)
				self.cv.delete(*tmpTag)
			# mbox.showinfo('', 'deleted')
			if (toID + 1) in self.tree.idDict and self.tree.idDict[toID + 1] is not None:
				updateFromNode = self.tree.idDict[toID + 1]
				self._drawNodesAndLines(fromNode = updateFromNode, toNode = self.tree._last)
			else:
				self._lastDrawNode = self.tree._last
		elif 'lastDeletedIDs' in kw:
			for id in kw['lastDeletedIDs']:
				tmpTag = 'node'+str(id), str(id)+'->', '->'+str(id)
				self.cv.delete(*tmpTag)
		# mbox.showinfo('', 'drawed')

	def updateDrawing(self, operation, **kw):
		'''operation: 'append', 'delete', 'insert', 'redraw'. kw: fromID, toID, ID, fromNode, toNode, node, insertID  '''
		if operation == 'append':
			mbox.showinfo('', 'appending:')
			if self._lastDrawNode.next is None:
				mbox.showinfo('', 'nothing appended.')
			else:
				if self._initTreeSize == (self.tree.width, self.tree.height):
					self._drawNodesAndLines(fromNode = self._lastDrawNode.next, toNode = self.tree._last)
				else: 
					self._redraw()
			mbox.showinfo('', 'appended')
		if operation == 'delete':
			''' args is the same to delete in tree class. '''
			if 'fromID' in kw and 'toID' in kw:
				mbox.showinfo('', ('delete:', kw['fromID'], kw['toID']))
			elif 'ID' in kw:
				mbox.showinfo('', ('delete:', kw['ID']))
			else:
				mbox.showinfo('', 'delete last')
			if self._lastDrawNode == self.tree._last:
				mbox.showinfo('', 'nothing deleted.')
			else:
				fromNode = toNode = self._lastDrawNode
				fromID, toID = fromNode.nodeID, toNode.nodeID
				if 'node' in kw:
					fromID = toID = kw['node'].nodeID
				elif 'ID' in kw:
					fromID = toID = kw['ID']
				if 'fromNode' in kw and 'toNode' in kw:
					fromID, toID = kw['fromNode'].nodeID, kw['toNode'].nodeID
				elif 'fromID' in kw and 'toID' in kw:
					fromID, toID = kw['fromID'], kw['toID']
				assert fromID is not None and toID is not None, 'error 404'
				self._deleteNodeAndLines(fromID = fromID, toID = toID)
		if operation == 'insert':
			if self._initTreeSize != (self.tree.width, self.tree.height):
					self._redraw()
					mbox.showinfo('', 'inserted')
			else:
				if 'insertID' in kw:
					insertID = kw['insertID']
					self._deleteNodeAndLines(downFromNode = self.tree.idDict[insertID].parent)
					self._drawNodesAndLines(downFromNode = self.tree.idDict[insertID].parent)
				else:
					self._deleteNodeAndLines(deleteAll = True)
					self._drawNodesAndLines(fromNode = self.tree._root, toNode = self.tree._last)
				mbox.showinfo('', 'inserted')
		if operation == 'deleteDown':
			if 'lastDeletedIDs' in kw:
				self._deleteNodeAndLines(lastDeletedIDs = kw['lastDeletedIDs'])
				mbox.showinfo('', 'delete down:')
		if operation == 'redraw':
			self._redraw()
		
	def _redraw(self):
		self._initTreeSize = (self.tree.width, self.tree.height)
		self.root.destroy()
		self.__init__(self.tree)
		# self._drawCanvas()
		# self._drawNodesAndLines(fromNode = self.tree._root, toNode = self.tree._last)
		# mbox.showinfo('',('redrawed', self.tree.size))
		# self.root.mainloop()

class DrawTreeByList(DrawTreeByLink):
	def __init__(self, tree):
		super().__init__(tree)
	
	def _drawNodesAndLines(self, **kw):
		# draw nodes:
		for curNode in self.tree:
			if curNode.level == 1:
				curNode.drawXY = 0.5 * self.canvasWidth, self.offset
				curNode.drawWidth = self.canvasWidth
			else:
				w1 = curNode.parent.drawWidth
				w2 = curNode.drawWidth = w1 / (1 + max(curNode.parent.children.keys()))
				curNode.drawXY = curNode.parent.drawXY[0] - 1/2 * w1 + (1/2 + curNode.ndxInSib) * w2, \
								(curNode.level-1)*self.baseSize*self.zoomY + self.offset
			x1, y1 = curNode.drawXY
			x2, y2 = x1 + self.baseSize, y1 + self.baseSize
			tmpTag = ('nodes', 'node'+str(curNode.nodeID))
			self.cv.create_oval(x1, y1, x2, y2, fill = 'white', tag = tmpTag)
			self.cv.create_text(x1 + self.offset, y1 + self.offset, text = curNode.content, tag = tmpTag)

		#draw lines:
		for curNode in self.tree:
			for ch in curNode.children.values():
				x1, y1, x2, y2 = curNode.drawXY[0] + self.offset, curNode.drawXY[1] + self.offset,  \
					ch.drawXY[0] + self.offset, ch.drawXY[1] + self.offset
				tmpTag = ('lines', str(curNode.nodeID) + '->', '->'+ str(ch.nodeID))
				self.cv.create_line(x1, y1, x2, y2, width = 3, fill = 'red', tag = tmpTag)
			if curNode.parent is not None:   # to draw for newly added nodes.
				tmpTag = ('lines', str(curNode.parent.nodeID) + '->', '->' + str(curNode.nodeID))
				if self.cv.find('withtag', tmpTag) == ():
					parentXY = curNode.parent.drawXY[0] + self.offset, curNode.parent.drawXY[1] + self.offset
					x1, y1, x2, y2 =  parentXY[0], parentXY[1], 	\
						curNode.drawXY[0] + self.offset, curNode.drawXY[1] + self.offset
					self.cv.create_line(x1, y1, x2, y2, width = 3, fill = 'red', tag = tmpTag)					

		self.cv.lift('nodes')

		mbox.showinfo('','drawed')












