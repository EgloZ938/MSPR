const express = require('express');
const { Pool } = require('pg');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.static('public'));

// Configuration PostgreSQL
const pool = new Pool({
    user: 'egloz',  // votre nom d'utilisateur PostgreSQL
    host: 'localhost',
    database: 'covid_db',
    password: 'caca123',
    port: 5432,
});

// Route pour servir la page HTML
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public/index.html'));
});

// API pour récupérer toutes les données
app.get('/api/all-data', async (req, res) => {
    try {
        const result = await pool.query('SELECT * FROM covid_stats');
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

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Serveur démarré sur http://localhost:${PORT}`);
});