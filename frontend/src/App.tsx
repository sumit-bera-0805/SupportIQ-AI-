import './App.css'
import { useEffect, useMemo, useState } from 'react'
import { checkApiHealth } from './api'
import {
  Activity, ArrowDownRight, ArrowUpRight, Bell, Bot, BrainCircuit,
  ChartNoAxesCombined, ChevronDown, CircleHelp, Clock3,
  Download, Filter, LayoutDashboard, LifeBuoy, Menu, MessageSquare,
  Search, Settings2, ShieldAlert, Sparkles, X, Zap,
} from 'lucide-react'
import {
  Area, AreaChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'

const ticketTrend = [
  { day: 'Mon', tickets: 145, resolved: 112 },
  { day: 'Tue', tickets: 188, resolved: 156 },
  { day: 'Wed', tickets: 164, resolved: 140 },
  { day: 'Thu', tickets: 225, resolved: 185 },
  { day: 'Fri', tickets: 196, resolved: 172 },
  { day: 'Sat', tickets: 132, resolved: 119 },
  { day: 'Sun', tickets: 174, resolved: 151 },
]

const categoryData = [
  { name: 'Technical', value: 35, color: '#60a5fa' },
  { name: 'Billing', value: 24, color: '#a78bfa' },
  { name: 'Account', value: 19, color: '#34d399' },
  { name: 'Shipping', value: 14, color: '#fbbf24' },
  { name: 'Other', value: 8, color: '#64748b' },
]

const initialTickets = [
  { id: 'TK-2048', subject: 'Unable to access my account', category: 'Account', priority: 'High', status: 'Open', time: '4 min ago' },
  { id: 'TK-2047', subject: 'Payment deducted twice', category: 'Billing', priority: 'Critical', status: 'Open', time: '12 min ago' },
  { id: 'TK-2046', subject: 'Application crashes on startup', category: 'Technical', priority: 'High', status: 'In progress', time: '18 min ago' },
  { id: 'TK-2045', subject: 'Order delivery status update', category: 'Shipping', priority: 'Low', status: 'Resolved', time: '26 min ago' },
  { id: 'TK-2044', subject: 'Unable to reset password', category: 'Account', priority: 'Medium', status: 'In progress', time: '32 min ago' },
]

const navigation = [
  { name: 'Overview', icon: LayoutDashboard },
  { name: 'Ticket Explorer', icon: MessageSquare },
  { name: 'AI Analyzer', icon: BrainCircuit },
  { name: 'SLA Intelligence', icon: ShieldAlert },
  { name: 'Forecasting', icon: ChartNoAxesCombined },
  { name: 'Similar Incidents', icon: Search },
  { name: 'Model Insights', icon: Activity },
]

function App() {
  const [activePage, setActivePage] = useState('Overview')
  const [query, setQuery] = useState('')
  const [priorityFilter, setPriorityFilter] = useState('All')
  const [darkMode, setDarkMode] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [analyzerText, setAnalyzerText] = useState('')
  const [analyzerResult, setAnalyzerResult] = useState('')
  const [showNotifications, setShowNotifications] = useState(false)



  const [apiStatus, setApiStatus] = useState<
    'checking' | 'online' | 'offline'
  >('checking')

  useEffect(() => {
    let mounted = true

    const checkConnection = async () => {
      try {
        const result = await checkApiHealth()
        if (mounted) {
          setApiStatus(result.status === 'healthy' ? 'online' : 'offline')
        }
      } catch {
        if (mounted) setApiStatus('offline')
      }
    }

    void checkConnection()

    return () => {
      mounted = false
    }
  }, [])

  const filteredTickets = useMemo(() => initialTickets.filter((ticket) => {
    const matchesQuery = `${ticket.id} ${ticket.subject} ${ticket.category}`
      .toLowerCase().includes(query.toLowerCase())
    const matchesPriority = priorityFilter === 'All' || ticket.priority === priorityFilter
    const matchesPage = activePage !== 'Ticket Explorer' || true
    return matchesQuery && matchesPriority && matchesPage
  }), [query, priorityFilter, activePage])

  const exportTickets = () => {
    const rows = filteredTickets.map((ticket) =>
      [ticket.id, ticket.subject, ticket.category, ticket.priority, ticket.status]
        .map((value) => `"${value.replaceAll('"', '""')}"`).join(',')
    )
    
const headers = ['Ticket ID', 'Subject', 'Category', 'Priority', 'Status']
const blob = new Blob([[headers.join(','), ...rows].join('\n')], {
  type: 'text/csv;charset=utf-8;',
})

    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'supportiq-tickets.csv'
    link.click()
    URL.revokeObjectURL(url)
  }

  const runDemoAnalysis = () => {
    if (!analyzerText.trim()) {
      setAnalyzerResult('Please enter a ticket description first.')
      return
    }
    setAnalyzerResult('Demo preview only — connect the FastAPI prediction endpoint to run a real ML prediction.')
  }

  const isDark = darkMode

  return (
    <div className={isDark ? 'app-shell dark-theme' : 'app-shell light-theme'}>
      {sidebarOpen && <button aria-label="Close navigation" className="mobile-overlay" onClick={() => setSidebarOpen(false)} />}
      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-mark"><BrainCircuit size={23} /></div>
          <div><strong>SupportIQ<span>.AI</span></strong><small>INTELLIGENCE PLATFORM</small></div>
          <button className="icon-button mobile-close" aria-label="Close sidebar" onClick={() => setSidebarOpen(false)}><X size={18} /></button>
        </div>

        <div className="workspace-switch">
          <div className="workspace-avatar">S</div>
          <div className="workspace-copy"><strong>Support Workspace</strong><small>Personal workspace</small></div>
          <ChevronDown size={15} />
        </div>

        <p className="nav-label">WORKSPACE</p>
        <nav className="nav-list">
          {navigation.map(({ name, icon: Icon }) => (
             <button key={name} className={`nav-item ${activePage === name ? 'active' : ''}`}
              onClick={() => { setActivePage(name); setSidebarOpen(false) }}>
              <Icon size={18} strokeWidth={1.8} /><span>{name}</span>
              {name === 'AI Analyzer' && <span className="new-badge">AI</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="plan-card">
            <div className="plan-icon"><Sparkles size={17} /></div>
            <strong>Intelligence engine</strong>
            <p>Explore automated ticket insights and risk signals.</p>
            <button onClick={() => setActivePage('Model Insights')}>Explore insights <span>→</span></button>
          </div>
          <button className="nav-item" onClick={() => setDarkMode(!darkMode)}>
            <Settings2 size={18} /><span>Appearance</span><span className="theme-indicator">{darkMode ? 'Dark' : 'Light'}</span>
          </button>
          <div className="profile-row"><div className="profile-avatar">SB</div><div><strong>Sumit Bera</strong><small>Project administrator</small></div><ChevronDown size={15} /></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}><Menu size={20} /></button>
            <div className="breadcrumb">Workspace <span>/</span> <strong>{activePage}</strong></div>
          </div>
          <div className="topbar-actions">
            
<div className={`system-status api-${apiStatus}`}>
  <span />
  {apiStatus === 'checking'
    ? 'Checking API...'
    : apiStatus === 'online'
      ? 'API connected'
      : 'API unavailable'}
</div>

            <button className="icon-button notification-button" aria-label="Notifications" onClick={() => setShowNotifications(!showNotifications)}><Bell size={18} /><i /></button>
            <button className="help-button" onClick={() => window.alert('SupportIQ AI — customer support intelligence demo.') }><CircleHelp size={17} /> Help</button>
          </div>
          {showNotifications && <div className="notification-popover"><strong>Notifications</strong><p>No new notifications.</p></div>}
        </header>

        <div className="page-content">
          <section className="welcome-row">
            <div>
              <div className="eyebrow"><span className="eyebrow-dot" /> CUSTOMER SUPPORT INTELLIGENCE</div>
              <h1>{activePage === 'Overview' ? 'Support overview' : activePage}<span className="title-period">.</span></h1>
              <p className="page-subtitle">Monitor performance, uncover patterns, and turn support data into action.</p>
            </div>
            <div className="welcome-actions">
              <span className="date-chip"><Clock3 size={15} /> Last 7 days <ChevronDown size={14} /></span>
              <button className="primary-button" onClick={exportTickets}><Download size={16} /> Export report</button>
            </div>
          </section>

          <div className="demo-notice"><Sparkles size={16} /><span><strong>Interactive demo data</strong> — sample metrics are illustrative, not live customer data.</span><span className="demo-pill">DEMO MODE</span></div>

          <section className="metrics-grid">
            <MetricCard title="Total tickets" value="12,486" change="+12.8%" positive={true} icon={MessageSquare} accent="blue" note="vs. previous period" />
            <MetricCard title="Open tickets" value="1,284" change="+4.2%" positive={false} icon={LifeBuoy} accent="violet" note="vs. previous period" />
            <MetricCard title="SLA compliance" value="94.8%" change="+2.4%" positive={true} icon={ShieldAlert} accent="green" note="vs. previous period" />
            <MetricCard title="Avg. response" value="18.6m" change="-8.1%" positive={true} icon={Zap} accent="amber" note="vs. previous period" />
          </section>

           {activePage === 'AI Analyzer' && (
            <section className="panel analyzer-panel">
              <PanelHeading icon={Bot} title="AI ticket analyzer" subtitle="Enter a support issue to preview the analysis workflow." />
              <label className="field-label" htmlFor="ticket-description">Ticket description</label>
              <textarea id="ticket-description" value={analyzerText} onChange={(event) => setAnalyzerText(event.target.value)} placeholder="Example: I was charged twice for my subscription..." rows={5} />
              <button className="primary-button analyze-button" onClick={runDemoAnalysis}><Sparkles size={16} /> Analyze ticket</button>
              {analyzerResult && <div className="analysis-result" role="status">{analyzerResult}</div>}
            </section>
          )}

          {activePage === 'Model Insights' && (
            <section className="panel">
              <PanelHeading icon={BrainCircuit} title="Model intelligence" subtitle="Evaluation metrics will be connected to your trained model." />
              <div className="empty-state"><Activity size={28} /><strong>Model evaluation workspace</strong><p>Connect the FastAPI model metrics endpoint to display real macro-F1, precision, recall, and confusion matrix results.</p></div>
            </section>
          )}

          {activePage === 'SLA Intelligence' && (
            <section className="panel">
              <PanelHeading icon={ShieldAlert} title="SLA risk intelligence" subtitle="Monitor deadline risk when valid SLA timestamps are available." />
              <div className="risk-summary"><div><small>Risk rules</small><strong>Awaiting API</strong></div><div><small>Deadline analysis</small><strong>Not connected</strong></div><div><small>Prediction source</small><strong>Backend required</strong></div></div>
            </section>
          )}

          {activePage === 'Forecasting' && (
            <section className="panel">
              <PanelHeading icon={ChartNoAxesCombined} title="Ticket volume forecasting" subtitle="Historical counts and forecasts will come from the Python analytics API." />
              <div className="empty-state"><ChartNoAxesCombined size={28} /><strong>Forecasting workspace</strong><p>Connect the forecasting endpoint to show actual historical data, forecast horizons, and backtesting metrics.</p></div>
            </section>
          )}

          {activePage === 'Similar Incidents' && (
            <section className="panel">
              <PanelHeading icon={Search} title="Similar incident search" subtitle="Find related support cases using the NLP similarity engine." />
              <div className="search-field wide-search"><Search size={17} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search sample tickets by ID, subject, or category..." /></div>
              <p className="subtle-copy">The current search filters the sample ticket table below. Similarity ranking will be connected to the API in a later step.</p>
            </section>
          )}

          <div className="dashboard-grid">
            <section className="panel trend-panel">
              <PanelHeading icon={ChartNoAxesCombined} title="Ticket activity" subtitle="Incoming tickets and resolutions over the last 7 days."
                action={<button className="select-button" onClick={() => setPriorityFilter(priorityFilter === 'All' ? 'High' : 'All')}>{priorityFilter === 'All' ? 'All priorities' : 'High priority'} <ChevronDown size={14} /></button>} />
              <div className="chart-legend"><span><i className="legend-blue" /> Incoming</span><span><i className="legend-violet" /> Resolved</span></div>
              <div className="trend-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={ticketTrend} margin={{ top: 12, right: 8, left: -18, bottom: 0 }}>
                    <defs>
                      <linearGradient id="incomingGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#60a5fa" stopOpacity={0.28} /><stop offset="95%" stopColor="#60a5fa" stopOpacity={0} /></linearGradient>
                      <linearGradient id="resolvedGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#a78bfa" stopOpacity={0.2} /><stop offset="95%" stopColor="#a78bfa" stopOpacity={0} /></linearGradient>
                    </defs>
                    <CartesianGrid stroke="#263247" strokeDasharray="3 5" vertical={false} />
                    <XAxis dataKey="day" tick={{ fill: '#8b9bb5', fontSize: 12 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: '#8b9bb5', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: '#111a2a', border: '1px solid #263247', borderRadius: 12, color: '#e8eefc' }} />
                    <Area type="monotone" dataKey="tickets" name="Incoming" stroke="#60a5fa" strokeWidth={2.5} fill="url(#incomingGradient)" />
                    <Area type="monotone" dataKey="resolved" name="Resolved" stroke="#a78bfa" strokeWidth={2.5} fill="url(#resolvedGradient)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="chart-footer"><span><Activity size={15} /> Activity updates when connected to your dataset</span><span className="live-indicator"><i /> Sample</span></div>
            </section>

              <section className="panel category-panel">
              <PanelHeading icon={Sparkles} title="Ticket categories" subtitle="Distribution across sample ticket types." />
              <div className="category-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={categoryData} dataKey="value" nameKey="name" innerRadius={66} outerRadius={92} paddingAngle={4} stroke="none">
                      {categoryData.map((item) => <Cell key={item.name} fill={item.color} />)}
                    </Pie>
                    <Tooltip contentStyle={{ background: '#111a2a', border: '1px solid #263247', borderRadius: 12, color: '#e8eefc' }} formatter={(value) => [`${value}%`, 'Share']} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="donut-center"><strong>100%</strong><span>of tickets</span></div>
              </div>
              <div className="category-legend">
                {categoryData.map((item) => <div className="category-legend-row" key={item.name}><span><i style={{ background: item.color }} />{item.name}</span><strong>{item.value}%</strong></div>)}
              </div>
            </section>
          </div>

          <section className="panel tickets-panel">
            <PanelHeading icon={MessageSquare} title="Recent tickets" subtitle="Explore the latest sample support requests."
              action={<button className="text-button" onClick={() => setActivePage('Ticket Explorer')}>View all tickets <span>→</span></button>} />
            <div className="table-toolbar">
              <div className="search-field"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search tickets..." /></div>
              <button className="filter-button" onClick={() => setPriorityFilter(priorityFilter === 'All' ? 'High' : 'All')}><Filter size={15} /> {priorityFilter === 'All' ? 'Filter' : 'High only'} <ChevronDown size={14} /></button>
            </div>
            <div className="table-scroll">
              <table>
                <thead><tr><th>Ticket</th><th>Category</th><th>Priority</th><th>Status</th><th>Created</th></tr></thead>
                <tbody>
                  {filteredTickets.map((ticket) => (
                    <tr key={ticket.id}>
                      <td><div className="ticket-cell"><span className="ticket-symbol"><MessageSquare size={15} /></span><div><strong>{ticket.id}</strong><small>{ticket.subject}</small></div></div></td>
                      <td><span className="category-tag">{ticket.category}</span></td>
                      <td><span className={`priority-tag priority-${ticket.priority.toLowerCase()}`}><i />{ticket.priority}</span></td>
                      <td><span className={`status-tag status-${ticket.status.toLowerCase().replace(' ', '-')}`}>{ticket.status}</span></td>
                      <td className="time-cell">{ticket.time}</td>
                    </tr>
                  ))}
                  {filteredTickets.length === 0 && <tr><td colSpan={5} className="empty-table">No tickets match these filters.</td></tr>}
                </tbody>
              </table>
            </div>
            <div className="table-footer"><span>Showing {filteredTickets.length} sample tickets</span><span>Sample data · Not connected to live records</span></div>
          </section>

          <footer className="page-footer"><span>© 2026 SupportIQ AI · Portfolio project</span><span><span className="footer-status" /> Demo environment <span className="footer-separator">·</span> Built for responsible AI</span></footer>
        </div>
      </main>
    </div>
  )
}

function MetricCard({
  title,
  value,
  change,
  positive,
  icon: Icon,
  accent,
  note,
  }: {
  title: string
  value: string
  change: string
  positive: boolean
  icon: React.ElementType
  accent: string
  note?: string
}) {
  return (
    <article className="metric-card">
      <div className="metric-top"><span>{title}</span><span className={`metric-icon icon-${accent}`}><Icon size={19} /></span></div>
      <div className="metric-value">{value}</div>
      <div className="metric-bottom"><span className={positive ? 'change-positive' : 'change-negative'}>{positive ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}{change}</span><span>{note}</span></div>
      <div className={`metric-accent accent-${accent}`} />
    </article>
  )
}




function PanelHeading({
  icon: Icon,
  title,
  subtitle,
  action,
}: {
  icon: React.ElementType
  title: string
  subtitle: string
  action?: React.ReactNode
}) {
  return (
    <div className="panel-heading">
      <div className="panel-title-wrap">
        <span className="panel-icon"><Icon size={17} /></span>
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>
      {action && <div className="panel-action">{action}</div>}
    </div>
  )
}

export default App

                  



