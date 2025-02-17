import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const Dashboard = () => {
    const navigate = useNavigate();
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);

    const handleLogout = () => {
        localStorage.removeItem('token');
        navigate('/'); // Redirect to the login page after logout
    };

    const toggleDropdown = () => {
        setIsDropdownOpen(!isDropdownOpen);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black">
            {/* Navbar */}
            <nav className="bg-black/30 backdrop-blur-sm px-4 py-4 fixed w-full top-0 z-10 shadow-lg">
                <div className="container mx-auto flex items-center justify-between">
                    <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600">
                        Dashboard
                    </h1>
                    <div className="relative">
                        <button
                            onClick={toggleDropdown}
                            className="flex items-center space-x-2 bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 text-white px-4 py-2 rounded-lg hover:scale-105 transform transition-all"
                        >
                            <span className="hidden sm:inline">Profile</span>
                            <svg
                                className={`w-5 h-5 transform transition-transform ${
                                    isDropdownOpen ? 'rotate-180' : ''
                                }`}
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth="2"
                                    d="M19 9l-7 7-7-7"
                                ></path>
                            </svg>
                        </button>
                        {isDropdownOpen && (
                            <div className="absolute right-0 mt-2 w-48 bg-gray-800 rounded-lg shadow-xl z-20">
                                <button
                                    className="block w-full text-left px-4 py-2 text-gray-100 hover:bg-gray-700"
                                    onClick={() => navigate('/profile')}
                                >
                                    Profile
                                </button>
                                <button
                                    className="block w-full text-left px-4 py-2 text-gray-100 hover:bg-gray-700"
                                    onClick={handleLogout}
                                >
                                    Logout
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <div className="pt-20 container mx-auto px-4">
                <h2 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600 mb-8">
                    Welcome to Your Dashboard
                </h2>
                <p className="text-lg text-gray-200 mb-4">
                    Here you can manage your account, view your activities, and more.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                    {/* Example Cards */}
                    <div className="bg-black/30 backdrop-blur-sm p-6 rounded-xl shadow-lg hover:scale-105 transform transition-all">
                        <h3 className="text-xl font-bold text-purple-400">Card Title</h3>
                        <p className="text-gray-200 mt-2">This is some example content.</p>
                    </div>
                    <div className="bg-black/30 backdrop-blur-sm p-6 rounded-xl shadow-lg hover:scale-105 transform transition-all">
                        <h3 className="text-xl font-bold text-purple-400">Card Title</h3>
                        <p className="text-gray-200 mt-2">This is some example content.</p>
                    </div>
                    <div className="bg-black/30 backdrop-blur-sm p-6 rounded-xl shadow-lg hover:scale-105 transform transition-all">
                        <h3 className="text-xl font-bold text-purple-400">Card Title</h3>
                        <p className="text-gray-200 mt-2">This is some example content.</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
