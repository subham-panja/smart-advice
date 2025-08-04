'use client';

import { useState } from 'react';
import { CogIcon, BellIcon, UserIcon, ShieldCheckIcon, SunIcon, MoonIcon, ComputerDesktopIcon } from '@heroicons/react/24/outline';
import { useTheme } from '../contexts/ThemeContext';

interface SettingsSection {
  id: string;
  name: string;
  icon: React.ComponentType<{ className?: string }>;
}

const settingsSections: SettingsSection[] = [
  { id: 'general', name: 'General', icon: CogIcon },
  { id: 'notifications', name: 'Notifications', icon: BellIcon },
  { id: 'profile', name: 'Profile', icon: UserIcon },
  { id: 'privacy', name: 'Privacy & Security', icon: ShieldCheckIcon },
];

export default function Settings() {
  const { theme, setTheme } = useTheme();
  const [activeSection, setActiveSection] = useState('general');
  const [settings, setSettings] = useState({
    general: {
      autoRefresh: true,
      refreshInterval: 300,
      defaultView: 'recommendations',
    },
    notifications: {
      emailAlerts: true,
      pushNotifications: false,
      weeklyReports: true,
    },
    profile: {
      name: '',
      email: '',
      riskTolerance: 'moderate',
    },
    privacy: {
      dataSharing: false,
      analyticsTracking: true,
    },
  });

  const updateSetting = (section: string, key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section as keyof typeof prev],
        [key]: value,
      },
    }));
  };

  const renderGeneralSettings = () => (
    <div className="space-y-6">
      <div>
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={settings.general.autoRefresh}
            onChange={(e) => updateSetting('general', 'autoRefresh', e.target.checked)}
            className="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
          <span className="text-gray-900 dark:text-gray-100">Auto-refresh data</span>
        </label>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 ml-7">
          Automatically refresh stock data and recommendations
        </p>
      </div>
      
      <div>
        <label className="block text-gray-900 dark:text-gray-100 mb-2">
          Refresh Interval (seconds)
        </label>
        <select
          value={settings.general.refreshInterval}
          onChange={(e) => updateSetting('general', 'refreshInterval', parseInt(e.target.value))}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
        >
          <option value={60}>1 minute</option>
          <option value={300}>5 minutes</option>
          <option value={600}>10 minutes</option>
          <option value={1800}>30 minutes</option>
        </select>
      </div>

      <div>
        <label className="block text-gray-900 dark:text-gray-100 mb-2">
          Default View
        </label>
        <select
          value={settings.general.defaultView}
          onChange={(e) => updateSetting('general', 'defaultView', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
        >
          <option value="dashboard">Dashboard</option>
          <option value="analysis">Generate Analysis</option>
          <option value="recommendations">View Recommendations</option>
        </select>
      </div>

      <div>
        <label className="block text-gray-900 dark:text-gray-100 mb-3">
          Theme
        </label>
        <div className="grid grid-cols-3 gap-3">
          <button
            onClick={() => setTheme('light')}
            className={`flex flex-col items-center p-3 rounded-lg border-2 transition-colors ${
              theme === 'light'
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500 text-gray-700 dark:text-gray-300'
            }`}
          >
            <SunIcon className="h-6 w-6 mb-2" />
            <span className="text-sm font-medium">Light</span>
          </button>
          
          <button
            onClick={() => setTheme('dark')}
            className={`flex flex-col items-center p-3 rounded-lg border-2 transition-colors ${
              theme === 'dark'
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500 text-gray-700 dark:text-gray-300'
            }`}
          >
            <MoonIcon className="h-6 w-6 mb-2" />
            <span className="text-sm font-medium">Dark</span>
          </button>
          
          <button
            onClick={() => setTheme('system')}
            className={`flex flex-col items-center p-3 rounded-lg border-2 transition-colors ${
              theme === 'system'
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500 text-gray-700 dark:text-gray-300'
            }`}
          >
            <ComputerDesktopIcon className="h-6 w-6 mb-2" />
            <span className="text-sm font-medium">System</span>
          </button>
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
          Choose your preferred theme. System will follow your device settings.
        </p>
      </div>
    </div>
  );

  const renderNotificationSettings = () => (
    <div className="space-y-6">
      <div>
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={settings.notifications.emailAlerts}
            onChange={(e) => updateSetting('notifications', 'emailAlerts', e.target.checked)}
            className="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
          <span className="text-gray-900 dark:text-gray-100">Email alerts</span>
        </label>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 ml-7">
          Receive email notifications for important updates
        </p>
      </div>

      <div>
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={settings.notifications.pushNotifications}
            onChange={(e) => updateSetting('notifications', 'pushNotifications', e.target.checked)}
            className="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
          <span className="text-gray-900 dark:text-gray-100">Push notifications</span>
        </label>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 ml-7">
          Get instant notifications in your browser
        </p>
      </div>

      <div>
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={settings.notifications.weeklyReports}
            onChange={(e) => updateSetting('notifications', 'weeklyReports', e.target.checked)}
            className="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
          <span className="text-gray-900 dark:text-gray-100">Weekly reports</span>
        </label>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 ml-7">
          Receive weekly summary of your portfolio performance
        </p>
      </div>
    </div>
  );

  const renderProfileSettings = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-gray-900 dark:text-gray-100 mb-2">
          Name
        </label>
        <input
          type="text"
          value={settings.profile.name}
          onChange={(e) => updateSetting('profile', 'name', e.target.value)}
          placeholder="Enter your name"
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
        />
      </div>

      <div>
        <label className="block text-gray-900 dark:text-gray-100 mb-2">
          Email
        </label>
        <input
          type="email"
          value={settings.profile.email}
          onChange={(e) => updateSetting('profile', 'email', e.target.value)}
          placeholder="Enter your email"
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
        />
      </div>

      <div>
        <label className="block text-gray-900 dark:text-gray-100 mb-2">
          Risk Tolerance
        </label>
        <select
          value={settings.profile.riskTolerance}
          onChange={(e) => updateSetting('profile', 'riskTolerance', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
        >
          <option value="conservative">Conservative</option>
          <option value="moderate">Moderate</option>
          <option value="aggressive">Aggressive</option>
        </select>
      </div>
    </div>
  );

  const renderPrivacySettings = () => (
    <div className="space-y-6">
      <div>
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={settings.privacy.dataSharing}
            onChange={(e) => updateSetting('privacy', 'dataSharing', e.target.checked)}
            className="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
          <span className="text-gray-900 dark:text-gray-100">Share data for research</span>
        </label>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 ml-7">
          Help improve our algorithms by sharing anonymized data
        </p>
      </div>

      <div>
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={settings.privacy.analyticsTracking}
            onChange={(e) => updateSetting('privacy', 'analyticsTracking', e.target.checked)}
            className="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
          <span className="text-gray-900 dark:text-gray-100">Analytics tracking</span>
        </label>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 ml-7">
          Allow us to track usage for improving user experience
        </p>
      </div>
    </div>
  );

  const renderSettingsContent = () => {
    switch (activeSection) {
      case 'general':
        return renderGeneralSettings();
      case 'notifications':
        return renderNotificationSettings();
      case 'profile':
        return renderProfileSettings();
      case 'privacy':
        return renderPrivacySettings();
      default:
        return renderGeneralSettings();
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-4">
          Settings
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Manage your preferences and account settings
        </p>
      </div>

      <div className="grid lg:grid-cols-4 gap-8">
        {/* Settings Navigation */}
        <div className="lg:col-span-1">
          <nav className="bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
            <ul className="divide-y divide-gray-200 dark:divide-gray-700">
              {settingsSections.map((section) => {
                const Icon = section.icon;
                return (
                  <li key={section.id}>
                    <button
                      onClick={() => setActiveSection(section.id)}
                      className={`w-full flex items-center px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors ${
                        activeSection === section.id
                          ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 border-r-2 border-blue-500'
                          : 'text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      <Icon className="h-5 w-5 mr-3" />
                      {section.name}
                    </button>
                  </li>
                );
              })}
            </ul>
          </nav>
        </div>

        {/* Settings Content */}
        <div className="lg:col-span-3">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 border border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-6">
              {settingsSections.find(s => s.id === activeSection)?.name}
            </h2>
            {renderSettingsContent()}
            
            <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-600">
              <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">
                Save Changes
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
