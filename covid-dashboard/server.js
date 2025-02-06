const express = require('express');
const { Pool } = require('pg');
const cors = require('cors');
const path = require('path');
const jwt = require('jsonwebtoken');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configuration PostgreSQL avec variables d'environnement
const pool = new Pool({
    user: process.env.DB_USER,
    host: process.env.DB_HOST,
    database: process.env.DB_NAME,
    password: process.env.DB_PASSWORD,
    port: process.env.DB_PORT,
});

// Middleware d'authentification
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
        return res.status(401).json({ error: 'Token manquant' });
    }

    jwt.verify(token, process.env.JWT_SECRET || 'votre_secret_jwt', (err, user) => {
        if (err) {
            return res.status(403).json({ error: 'Token invalide' });
        }
        req.user = user;
        next();
    });
};

// Route de login (nouvelle)
app.post('/api/admin/login', (req, res) => {
    const { username, password } = req.body;

    if (username === process.env.ADMIN_USERNAME && 
        password === process.env.ADMIN_PASSWORD) {
        
        const token = jwt.sign(
            { username: process.env.ADMIN_USERNAME },
            process.env.JWT_SECRET || 'votre_secret_jwt',
            { expiresIn: '1h' }
        );

        res.json({ token });
    } else {
        res.status(401).json({ error: 'Identifiants incorrects' });
    }
});

// Route pour servir la page HTML
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public/index.html'));
});

// API pour récupérer toutes les données
app.get('/api/all-data', async (req, res) => {
    try {
        const result = await pool.query('SELECT * FROM covid_stats ORDER BY country_region');
        res.json(result.rows);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// API pour les statistiques globales
app.get('/api/global-stats', async (req, res) => {
    try {
        const result = await pool.query(`
            SELECT 
                SUM(confirmed) as total_confirmed,
                SUM(deaths) as total_deaths,
                SUM(recovered) as total_recovered,
                SUM(active) as total_active
            FROM covid_stats
        `);
        res.json(result.rows[0]);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// API pour les top 10 pays
app.get('/api/top-countries', async (req, res) => {
    try {
        const result = await pool.query(`
            SELECT country_region, confirmed, deaths, recovered
            FROM covid_stats
            ORDER BY confirmed DESC
            LIMIT 10
        `);
        res.json(result.rows);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Serveur démarré sur http://localhost:${PORT}`);
});