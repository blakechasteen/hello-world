import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Network,
  Activity,
  ChevronLeft,
  ChevronRight,
  Database,
  GitBranch,
  Zap
} from 'lucide-react'

const navItems = [
  {
    name: 'Dashboard',
    path: '/',
    icon: LayoutDashboard,
    description: 'System overview'
  },
  {
    name: 'Yarn Graph',
    path: '/graph',
    icon: Network,
    description: 'Knowledge graph'
  },
  {
    name: 'Monitoring',
    path: '/monitoring',
    icon: Activity,
    description: 'Metrics & alerts'
  },
]

const quickLinks = [
  { name: 'Memories', icon: Database },
  { name: 'Workflow', icon: GitBranch },
  { name: 'Performance', icon: Zap },
]

function Sidebar({ open, onToggle }) {
  const location = useLocation()

  return (
    <aside className={`bg-white border-r border-gray-200 transition-all duration-300 ${
      open ? 'w-64' : 'w-16'
    }`}>
      <div className="h-full flex flex-col">
        {/* Toggle Button */}
        <div className="p-4 border-b border-gray-200 flex justify-end">
          <button
            onClick={onToggle}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            {open ? (
              <ChevronLeft className="w-5 h-5 text-gray-600" />
            ) : (
              <ChevronRight className="w-5 h-5 text-gray-600" />
            )}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path

            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-hololoom-primary to-hololoom-secondary text-white shadow-md'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Icon className={`w-5 h-5 ${isActive ? 'text-white' : 'text-gray-600'}`} />
                {open && (
                  <div className="flex-1">
                    <p className={`text-sm font-medium ${isActive ? 'text-white' : 'text-gray-900'}`}>
                      {item.name}
                    </p>
                    <p className={`text-xs ${isActive ? 'text-white/80' : 'text-gray-500'}`}>
                      {item.description}
                    </p>
                  </div>
                )}
              </Link>
            )
          })}
        </nav>

        {/* Quick Links */}
        {open && (
          <div className="p-4 border-t border-gray-200">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
              Quick Links
            </p>
            <div className="space-y-2">
              {quickLinks.map((link) => {
                const Icon = link.icon
                return (
                  <button
                    key={link.name}
                    className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-gray-700 hover:bg-gray-100 transition-colors"
                  >
                    <Icon className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">{link.name}</span>
                  </button>
                )
              })}
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}

export default Sidebar
