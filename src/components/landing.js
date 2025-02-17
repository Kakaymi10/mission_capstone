import React from 'react';

const Landing = () => {
    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-center text-gray-100 p-5">
            <div className="max-w-4xl mx-auto mt-20 animate-fade-in-down">
                <h1 className="text-6xl mb-8 font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600 animate-pulse">
                    Smart Wallet Manager
                </h1>
                <p className="text-2xl max-w-3xl mx-auto mb-12 text-cyan-300 leading-relaxed animate-float">
                    Take control of your finances with our comprehensive wallet management solution.
                    Track expenses, monitor budgets, and visualize your financial journey.
                </p>
                <ul className="text-left mb-12 space-y-4">
                    {[
                        'Track all transactions across multiple accounts',
                        'Generate custom time-based reports',
                        'Set and monitor budget limits',
                        'Organize expenses with categories',
                        'Visualize your financial summary'
                    ].map((feature, index) => (
                        <li key={index} className="flex items-center text-lg transform hover:scale-105 transition-transform">
                            <span className="text-pink-500 mr-3 animate-bounce">★</span>
                            <span className="text-gradient-animated">{feature}</span>
                        </li>
                    ))}
                </ul>
                <a 
                    href="/signup"
                    className="inline-block px-12 py-5 text-2xl bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 text-white rounded-full 
                    hover:scale-110 transform transition-all duration-300 animate-glow shadow-xl hover:shadow-2xl
                    hover:shadow-purple-500/50"
                >
                    Get Started
                </a>
            </div>
        </div>
    );
};

export default Landing;
