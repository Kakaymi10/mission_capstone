import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css'; // Import Toast CSS

const InputField = ({ type, name, placeholder, value, onChange, required = true }) => (
    <div>
        <input
            type={type}
            name={name}
            placeholder={placeholder}
            value={value}
            onChange={onChange}
            className="w-full p-3 rounded-lg bg-gray-800/50 border border-gray-700 text-gray-100 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 transition-all"
            required={required}
        />
    </div>
);

const Signup = () => {
    const navigate = useNavigate();
    const [haveAccount, setHaveAccount] = useState(false);
    const [formData, setFormData] = useState({
        email: '',
        password: '',
        confirmPassword: '',
        name: '',
    });
    const [loading, setLoading] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);

        try {
            if (!haveAccount) {
                if (formData.password !== formData.confirmPassword) {
                    toast.error("Passwords don't match!");
                    setLoading(false);
                    return;
                }

                // Simulate signup API call
                await new Promise((resolve) => setTimeout(resolve, 1000)); // Mock delay
                toast.success('Account created successfully!');
                setHaveAccount(true);
            } else {
                const formUrlEncoded = new URLSearchParams();
                formUrlEncoded.append('username', formData.email);
                formUrlEncoded.append('password', formData.password);

                const response = await axios.post('http://127.0.0.1:8000/token', formUrlEncoded, {
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                });

                localStorage.setItem('token', response.data.access_token);
                toast.success('Login successful! Redirecting...');
                setTimeout(() => navigate('/dashboard'), 2000); // Short delay for user to see success message
                return;
            }
        } catch (error) {
            const errorMessage = error.response?.data?.detail || 'An unexpected error occurred.';
            toast.error(`Error: ${errorMessage}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black flex items-center justify-center p-5">
            <ToastContainer /> {/* Add ToastContainer */}
            <div className="w-full max-w-md">
                <div className="bg-black/30 backdrop-blur-sm p-8 rounded-2xl shadow-2xl">
                    <h2 className="text-4xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600">
                        {haveAccount ? 'Welcome Back' : 'Create Account'}
                    </h2>
                    <form onSubmit={handleSubmit} className="space-y-6">
                        {!haveAccount && (
                            <InputField
                                type="text"
                                name="name"
                                placeholder="Full Name"
                                value={formData.name}
                                onChange={handleChange}
                            />
                        )}
                        <InputField
                            type="email"
                            name="email"
                            placeholder="Email"
                            value={formData.email}
                            onChange={handleChange}
                        />
                        <InputField
                            type="password"
                            name="password"
                            placeholder="Password"
                            value={formData.password}
                            onChange={handleChange}
                        />
                        {!haveAccount && (
                            <InputField
                                type="password"
                                name="confirmPassword"
                                placeholder="Confirm Password"
                                value={formData.confirmPassword}
                                onChange={handleChange}
                            />
                        )}
                        <button
                            type="submit"
                            className="w-full py-3 bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 text-white rounded-lg
                            hover:scale-105 transform transition-all duration-300 shadow-xl hover:shadow-purple-500/50"
                            disabled={loading}
                        >
                            {loading ? (
                                <div className="flex items-center justify-center">
                                    <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path
                                            className="opacity-75"
                                            fill="currentColor"
                                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                        ></path>
                                    </svg>
                                    Processing...
                                </div>
                            ) : haveAccount ? 'Sign In' : 'Sign Up'}
                        </button>
                    </form>
                    <p className="mt-4 text-gray-400">
                        {haveAccount ? "Don't have an account?" : "Already have an account?"}
                        <button
                            onClick={() => setHaveAccount(!haveAccount)}
                            className="ml-2 text-purple-400 hover:text-purple-300 transition-colors"
                        >
                            {haveAccount ? 'Sign Up' : 'Sign In'}
                        </button>
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Signup;
