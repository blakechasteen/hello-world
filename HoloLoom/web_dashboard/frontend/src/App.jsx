import React, { useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import Header from './components/Layout/Header'
import Sidebar from './components/Layout/Sidebar'
import DashboardView from './views/DashboardView'
import GraphView from './views/GraphView'
import MonitoringView from './views/MonitoringView'
import './App.css'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <Sidebar open={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />

        {/* Content Area */}
        <main className="flex-1 overflow-y-auto p-6">
          <Routes>
            <Route path="/" element={<DashboardView />} />
            <Route path="/graph" element={<GraphView />} />
            <Route path="/monitoring" element={<MonitoringView />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default App
